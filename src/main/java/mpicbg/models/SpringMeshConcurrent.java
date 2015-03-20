package mpicbg.models;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.janelia.alignment.Distributer;

import mpicbg.models.ErrorStatistic;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.SpringMesh;

public class SpringMeshConcurrent
{
	final static private int VIS_SIZE = 512;
	final static private DecimalFormat DECIMAL_FORMAT = new DecimalFormat();

	static protected void println( final String s ){ IJ.log( s ); }

	
	/**
	 * Optimize a {@link Collection} of connected {@link SpringMesh SpringMeshes}.
	 * 
	 * @deprecated Remains for reproducing legacy results
	 * 
	 * @param maxError do not accept convergence if error is > max_error
	 * @param maxIterations stop after that many iterations even if there was
	 *   no minimum found
	 * @param maxPlateauwidth convergence is reached if the average slope in
	 *   an interval of this size is 0.0 (in double accuracy).  This prevents
	 *   the algorithm from stopping at plateaus smaller than this value.
	 * 
	 */
	@Deprecated
	public static void optimizeMeshes2(
			final ArrayList< SpringMesh > meshes,
			final double maxError,
			final int maxIterations,
			final int maxPlateauwidth,
			final int threadsNum ) throws NotEnoughDataPointsException 
	{
		optimizeMeshes2( meshes, maxError, maxIterations, maxPlateauwidth, threadsNum, false );
	}
	
	/**
	 * Optimize a {@link Collection} of connected {@link SpringMesh SpringMeshes}.
	 * 
	 * @deprecated Remains for reproducing legacy results

	 * @param maxError do not accept convergence if error is > max_error
	 * @param maxIterations stop after that many iterations even if there was
	 *   no minimum found
	 * @param maxPlateauwidth convergence is reached if the average slope in
	 *   an interval of this size is 0.0 (in double accuracy).  This prevents
	 *   the algorithm from stopping at plateaus smaller than this value.
	 * 
	 */
	@Deprecated
	public static void optimizeMeshes2(
			final ArrayList< SpringMesh > meshes,
			final double maxError,
			final int maxIterations,
			final int maxPlateauwidth,
			final int threadsNum,
			final boolean visualize ) throws NotEnoughDataPointsException 
	{
		final ErrorStatistic observer = new ErrorStatistic( maxPlateauwidth + 1 );
		final ErrorStatistic singleMeshObserver = new ErrorStatistic( maxPlateauwidth + 1 );
		
		int i = 0;
		
		double force = 0;
		double maxForce = 0;
		double minForce = 0;
		
		boolean proceed = i < maxIterations;
		
		/* <visualization> */
		final ImageStack stackAnimation = new ImageStack( VIS_SIZE, VIS_SIZE );
		final ImagePlus impAnimation = new ImagePlus();
		/* </visualization> */
		
		// Create thread pool and partition the layers between the threads
		final ExecutorService exec = Executors.newFixedThreadPool( threadsNum );
		final ArrayList< Future< ? > > tasks = new ArrayList< Future< ? > >();
		final double[] threadMaxSpeed = new double[ threadsNum ];
		final double[] threadForce = new double[ threadsNum ];
		final double[] threadMeshMinForce = new double[ threadsNum ];
		final double[] threadMeshMaxForce = new double[ threadsNum ];

		println( "i mean min max" );
		
		while ( proceed )
		{
			force = 0;
			maxForce = 0;
			minForce = Double.MAX_VALUE;
			
			double maxSpeed = 0;
			
			/* <visualization> */
//			stackAnimation.addSlice( "" + i, paintMeshes( meshes, scale ) );
			if ( visualize )
			{
				stackAnimation.addSlice( "" + i, SpringMesh.paintSprings( meshes, VIS_SIZE, VIS_SIZE, maxError ) );
				impAnimation.setStack( stackAnimation );
				impAnimation.updateAndDraw();
				if ( i == 1 )
				{
					impAnimation.show();
				}
			}
			/* </visualization> */
			
			tasks.clear();
			
			Distributer distributer = new Distributer( meshes.size(), threadsNum );
			for ( int t = 0; t < threadsNum && distributer.hasNext(); t++ )
			{
				distributer.next();
				final int fromIndex = distributer.getStart();
				final int lastIndex = distributer.getEnd();
				final int threadIdx = t;

				tasks.add( exec.submit( new Runnable() {
					
					@Override
					public void run() {
						threadMaxSpeed[ threadIdx ] = 0;
						threadForce[ threadIdx ] = 0;
						threadMeshMaxForce[ threadIdx ] = 0;
						threadMeshMinForce[ threadIdx ] = Double.MAX_VALUE;
						
						for ( int meshIdx = fromIndex; meshIdx < lastIndex; meshIdx++ )
						{
							SpringMesh mesh = meshes.get( meshIdx );
							mesh.calculateForceAndSpeed( singleMeshObserver );
							threadForce[ threadIdx ] += mesh.getForce();
							if ( mesh.maxSpeed > threadMaxSpeed[ threadIdx ] )
								threadMaxSpeed[ threadIdx ] = mesh.maxSpeed;
							
							final double meshMaxForce = mesh.maxForce;
							final double meshMinForce = mesh.minForce;
							if ( meshMaxForce > threadMeshMaxForce[ threadIdx ] ) threadMeshMaxForce[ threadIdx ] = meshMaxForce;
							if ( meshMinForce < threadMeshMinForce[ threadIdx ] ) threadMeshMinForce[ threadIdx ] = meshMinForce;
						}
						
					}
				}));
			}
			
			// Join all threads, and collect their data
			int threadId = 0;
			for ( Future< ? > task : tasks )
			{
				try {
					
					task.get();
					force += threadForce[ threadId ];
					if ( threadMaxSpeed[ threadId ] > maxSpeed )
						maxSpeed = threadMaxSpeed[ threadId ];
					if ( threadMeshMaxForce[ threadId ] > maxForce ) maxForce = threadMeshMaxForce[ threadId ];
					if ( threadMeshMinForce[ threadId ] < minForce ) minForce = threadMeshMinForce[ threadId ];
				} catch (InterruptedException e) {
					exec.shutdownNow();
					e.printStackTrace();
				} catch (ExecutionException e) {
					exec.shutdownNow();
					e.printStackTrace();
				}
				threadId++;
			}
			
			observer.add( force / meshes.size() );
			
			final float dt = ( float )Math.min( 1000, 1.0 / maxSpeed );
			
			// Create mesh update threads
			tasks.clear();
			distributer = new Distributer( meshes.size(), threadsNum );
			for ( int t = 0; t < threadsNum && distributer.hasNext(); t++ )
			{
				distributer.next();
				final int fromIndex = distributer.getStart();
				final int lastIndex = distributer.getEnd();

				tasks.add( exec.submit( new Runnable() {
					
					@Override
					public void run() {
						for ( int meshIdx = fromIndex; meshIdx < lastIndex; meshIdx++ )
						{
							SpringMesh mesh = meshes.get( meshIdx );
							mesh.update( dt );
						}						
					}
				}));
			}
			
			// Join mesh update threads
			for ( Future< ? > task : tasks )
			{
				try {
					task.get();
				} catch (InterruptedException e) {
					exec.shutdownNow();
					e.printStackTrace();
				} catch (ExecutionException e) {
					exec.shutdownNow();
					e.printStackTrace();
				}
			}

			
			println( new StringBuffer( i + " " ).append( force / meshes.size() ).append( " " ).append( minForce ).append( " " ).append( maxForce ).toString() );
			
			if ( i > maxPlateauwidth )
			{
				proceed = force > maxError;
				
				int d = maxPlateauwidth;
				while ( !proceed && d >= 1 )
				{
					try
					{
						proceed |= Math.abs( observer.getWideSlope( d ) ) > 0.0;
					}
					catch ( final Exception e ) { e.printStackTrace(); }
					d /= 2;
				}
			}
			
			proceed &= ++i < maxIterations;
		}

		for ( final SpringMesh mesh : meshes )
		{
			mesh.updateAffines();
			mesh.updatePassiveVertices();
		}
		
		exec.shutdown();
		
		System.out.println( "Successfully optimized " + meshes.size() + " meshes after " + i + " iterations:" );
		System.out.println( "  average force: " + DECIMAL_FORMAT.format( force / meshes.size() ) + "N" );
		System.out.println( "  minimal force: " + DECIMAL_FORMAT.format( minForce ) + "N" );
		System.out.println( "  maximal force: " + DECIMAL_FORMAT.format( maxForce ) + "N" );
	}
	

}

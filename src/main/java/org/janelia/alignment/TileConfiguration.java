package org.janelia.alignment;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import mpicbg.models.ErrorStatistic;
import mpicbg.models.IllDefinedDataPointsException;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.PointMatch;
import mpicbg.models.Tile;
import mpicbg.models.TileUtil;

public class TileConfiguration extends mpicbg.models.TileConfiguration {

	protected int threadsNum;
	protected List< Tile< ? > > noApplyList;
	
	public TileConfiguration() {
		this.threadsNum = Runtime.getRuntime().availableProcessors();
		noApplyList = new ArrayList<Tile<?>>();
	}
	
	public void setThreadsNum( int threadsNum ) {
		this.threadsNum = threadsNum;
	}
	
	public void optimizeSilentlyConcurrent(
			final ErrorStatistic observer,
			final float maxAllowedError,
			final int maxIterations,
			final int maxPlateauwidth ) throws NotEnoughDataPointsException, IllDefinedDataPointsException, InterruptedException, ExecutionException 
	{
		TileUtil.optimizeConcurrently( observer, maxAllowedError, maxIterations, maxPlateauwidth,
				this, tiles, fixedTiles, threadsNum );
	}

	/**
	 * Estimate min/max/average displacement of all
	 * {@link PointMatch PointMatches} in all {@link Tile Tiles}.
	 */
	public void updateErrors( final ExecutorService exe, final int exeThreadsNum )
	{
		double cd = 0.0;
		minError = Double.MAX_VALUE;
		maxError = 0.0;
		
		final ArrayList< Future< ? > > futures = new ArrayList< Future< ? > >( exeThreadsNum );
		final double[] threadMinError = new double[ exeThreadsNum ];
		final double[] threadMaxError = new double[ exeThreadsNum ];
		final double[] threadCd = new double[ exeThreadsNum ];
		final Distributer distributer = new Distributer( tiles.size(), exeThreadsNum );
		final Tile< ? >[] tilesArr = tiles.toArray( new Tile< ? >[ tiles.size() ] );
		for ( int i = 0; i < exeThreadsNum && distributer.hasNext(); i++ )
		{
			distributer.next();
			final int threadIdx = i;
			threadMinError[ threadIdx ] = Double.MAX_VALUE;
			threadMaxError[ threadIdx ] = 0.0;
			threadCd[ threadIdx ] = 0.0;
			final int firstTileIdx = distributer.getStart();
			final int lastTileIdx = distributer.getEnd();
			futures.add(exe.submit( new Runnable() {
				
				@Override
				public void run() {
					for ( int idx = firstTileIdx; idx < lastTileIdx; idx++ ) {
						Tile< ? > t = tilesArr[ idx ];
						t.updateCost();
						final double d = t.getDistance();
						if ( d < threadMinError[ threadIdx ] ) threadMinError[ threadIdx ] = d;
						if ( d > threadMaxError[ threadIdx ] ) threadMaxError[ threadIdx ] = d;
						threadCd[ threadIdx ] += d;
					}
				}
			}));
		}
		
		for ( int i = 0; i < futures.size(); i++ )
		{
			try {
				futures.get(i).get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
			if ( threadMinError[ i ] < minError ) minError = threadMinError[ i ];
			if ( threadMaxError[ i ] > maxError ) maxError = threadMaxError[ i ];
			cd += threadCd[ i ];
			
		}

		cd /= tiles.size();
		error = cd;
	}

	
	public void addToNoApplyList( Tile< ? > t )
	{
		noApplyList.add( t );
	}

	// Override to avoid applying model on fixed tiles
	protected void apply()
	{
		for ( final Tile< ? > t : tiles )
		{
			if (! noApplyList.contains(t))
				t.apply();
		}
	}
}

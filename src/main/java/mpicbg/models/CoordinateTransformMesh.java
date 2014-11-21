package mpicbg.models;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Overrides the original CoordinateTransformMesh to also support multi-threaded
 * construction of the mesh
 * 
 * @author adisuis
 *
 */
public class CoordinateTransformMesh extends TransformMesh
{
	
	public CoordinateTransformMesh(
			final CoordinateTransform t,
			final int numX,
			final float width,
			final float height )
	{
		super( numX, numY( numX, width, height ), width, height );
		
		final Set< PointMatch > vertices = va.keySet();
		for ( final PointMatch vertex : vertices )
			vertex.getP2().apply( t );
		
		updateAffines();
	}
	
	public CoordinateTransformMesh(
			final CoordinateTransform t,
			final int numX,
			final float width,
			final float height,
			final int threadsNum )
	{
		super( numX, numY( numX, width, height ), width, height );
		
		final Set< PointMatch > vertices = va.keySet();
				
		
		final ExecutorService exec = Executors.newFixedThreadPool( threadsNum );
		final ArrayList< Future< ? > > tasks = new ArrayList< Future< ? > >();

		final int verticesPerThreadNum = vertices.size() / threadsNum;
		for ( int i = 0; i < threadsNum; i++ )
		{
			final int fromIndex = i * verticesPerThreadNum;
			final int lastIndex;
			if ( i == threadsNum - 1 ) // lastThread
				lastIndex = vertices.size();
			else
				lastIndex = fromIndex + verticesPerThreadNum;
			
			tasks.add( exec.submit( new Runnable() {
				
				@Override
				public void run() {
					final Iterator< PointMatch > it = vertices.iterator();
					int idx = 0;
					// Skip the initial part of the vertices
					while ( idx < fromIndex )
					{
						it.next();
						idx++;
					}
					// Apply the transformation to the relevant vertices
					while ( idx < lastIndex )
					{
						PointMatch vertex = it.next();
						vertex.getP2().apply( t );
						idx++;
					}
				}
			}));
		}
		
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
		exec.shutdown();
		
		updateAffines();
	}

	
}

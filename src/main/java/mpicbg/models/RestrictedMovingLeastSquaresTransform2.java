package mpicbg.models;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class RestrictedMovingLeastSquaresTransform2  extends AbstractMovingLeastSquaresTransform {

	protected static final float DEFAULT_NEIGHBORS_RADIUS = -1.0f;
	protected static final float EPS = 0.00001f;
	
	protected float neighborsRadius;
	/*
	private int numX; // num of points on the long row
	private int numY; // num of points on the long column
	*/

	@SuppressWarnings("rawtypes")
	protected ArrayList[][] buckets; // Assuming only a 2D grid
	protected static final int GRID_DIM = 16;
	protected int gridBlockDimSize[]; // the size (in pixels) of each dimension
	protected int maxBucketSize; // Holds the maximal number of points per bucket (memory allocation optimization)
	
	protected int[] deltaBlock;

	float[] meshBounds;

	
	protected float[][] p;
	protected float[][] q;
	protected float[] w;
	
	// Whether the data is ready for access by all threads
	private boolean dataReady;
	
	public RestrictedMovingLeastSquaresTransform2()
	{
		this(DEFAULT_NEIGHBORS_RADIUS);
	}

	public RestrictedMovingLeastSquaresTransform2( float neighborsDistance )
	{
		this.neighborsRadius = neighborsDistance;
		/*
		numX = -1;
		numY = -1;
		*/
		dataReady = false;
	}
	
	public void setRadius( float radius )
	{
		this.neighborsRadius = radius;
	}

	/*
	private static int comparePoints( float[][] arr, int idx1, int idx2 ) {
		if ( Math.abs(arr[0][idx1] - arr[0][idx2]) < EPS ) {
			// The X is equal, compare the Y
			if ( Math.abs(arr[1][idx1] - arr[1][idx2]) < EPS ) {
				return 0;
			}
			if ( arr[1][idx1] < arr[1][idx2] )
				return -1;
			return 1;
		}
		else if ( arr[0][idx1] < arr[0][idx2] )
			return -1;
		return 1;
	}

	private static void swap(float[] arr, int idx1, int idx2) {
		float tmp = arr[idx1];
		arr[idx1] = arr[idx2];
		arr[idx2] = tmp;
	}
	
	private static void sortPoints(float[][] arrP, float[][] arrQ, float[] arrW) {
		// bubble sort
		for ( int i = 0; i < arrP[0].length; i++ ) {
			for ( int j = i + 1; j < arrP[0].length; j++ ) {
				if ( comparePoints( arrP, i, j ) > 0 ) {
					// swap at indices i and j
					for ( int d = 0; d < arrP.length; d++ ) {
						swap( arrP[d], i, j );
						swap( arrQ[d], i, j );
					}
					swap( arrW, i, j );
				}
			}
		}
	}

	public void sortByP() {
		int dim = p.length;
		int n = p[0].length;
		
		// Sort all elements by their P value, so it will be easier to find the neighbors quickly
		sortPoints( p, q, w );
		
		if ( n == 2 ) {
			// compute the numX and numY values
			numY = 0;
			float minX = p[0][0];
			//The array is sorted, so only need to count all those with the same X at the beginning of the array
			for ( int i = 0; i < n; i++ ) {
				if ( Math.abs( p[0][i] - minX ) < EPS ) {
					
				}
				else
					break;
			}
		}
		else {
			throw new UnsupportedOperationException( "No implementation yet for something other than 2D data" );
		}
	}
	*/
	
	public float computeDefaultRadius() {
		// Estimates the radius by finding the minimal edge length between two points
		float minPointDistSqr = Float.MAX_VALUE;
		for ( int i = 0; i < p[0].length - 1; i++ ) {
			for ( int j = i + 1; j < p[0].length; j++ ) {
				float curDist = 0;
				for ( int d = 0; d < p.length; d++ ) {
					float delta = p[d][i] - p[d][j];
					curDist += delta * delta;
				}
				if ( curDist < minPointDistSqr )
					minPointDistSqr = curDist;
			}
		}
		return 2 * (float)Math.sqrt( minPointDistSqr );
	}
	
	private void calculateGridLocation( int i, int[] outDim ) {
		for ( int d = 0; d < outDim.length; d++ ) {
			outDim[ d ] = (int)( ( p[ d ][ i ] - meshBounds[ 2 * d ] ) / gridBlockDimSize[ d ] );
		}
	}

	private void calculateGridLocation( float[] location, int[] outDim ) {
		for ( int d = 0; d < outDim.length; d++ ) {
			outDim[ d ] = (int)( ( location[ d ] - meshBounds[ 2 * d ] ) / gridBlockDimSize[ d ] );
		}
	}

	@SuppressWarnings("unchecked")
	public synchronized void refreshGrid() {
		if ( dataReady == false ) {
			int dim = p.length;
			int n = p[0].length;
	
			if ( dim != 2 )
				throw new UnsupportedOperationException( "No implementation yet for a mesh other than 2D" );
	
			maxBucketSize = 0;
			
			meshBounds = new float[ 2 * dim ]; // minX maxX minY maxY [minZ maxZ]
			
			// Initialize the minimal and maximal values (x, y and possibly z) by using the first point
			for ( int d = 0; d < dim; d++ ) {
				meshBounds[ 2 * d ] = p[ d ][ 0 ];
				meshBounds[ 2 * d + 1 ] = p[ d ][ 0 ];
			}
	
			// Compute the max values across all points
			for ( int d = 0; d < dim; d++ ) {
				for ( int i = 1; i < n; i++ ) {
					meshBounds[ 2 * d ] = Math.min( meshBounds[ 2 * d ], p[ d ][ i ] );
					meshBounds[ 2 * d + 1 ] = Math.max( meshBounds[ 2 * d + 1 ], p[ d ][ i ] );
				}
			}
			System.out.print("meshBounds: ");
			for ( int d = 0; d < dim; d++ ) {
				System.out.print( meshBounds[ 2 * d ] + " " + meshBounds[ 2* d + 1 ] + " " );
			}
			System.out.println();
			
			// Compute the grid bucket dimensions sizes
			gridBlockDimSize = new int[ dim ];
			for ( int d = 0; d < dim; d++ ) {
				int maxValRounded = (int) Math.round( meshBounds[ 2 * d + 1 ] - meshBounds[ 2 * d ] + 0.5 );
				gridBlockDimSize[ d ] = (maxValRounded - 1) / GRID_DIM + 1;
				System.out.println( "gridBlockDimSize[" + d + "] = " + gridBlockDimSize[ d ] );
			}
			
			// Allocate the buckets for the grid and populate it with the data from the array p
			buckets = new ArrayList[GRID_DIM][GRID_DIM];
			for ( int blockX = 0; blockX < GRID_DIM; blockX++ ) {
				for ( int blockY = 0; blockY < GRID_DIM; blockY++ ) {
					buckets[ blockX ][ blockY ] = new ArrayList< Integer >();
				}
			}

			
			int[] outGridDim = new int[ dim ];
			for ( int i = 0; i < n; i++ ) {
				calculateGridLocation( i, outGridDim );
				buckets[ outGridDim[0] ][ outGridDim[1] ].add( i );
				if ( maxBucketSize < buckets[ outGridDim[0] ][ outGridDim[1] ].size() )
					maxBucketSize = buckets[ outGridDim[0] ][ outGridDim[1] ].size();
			}

			// Find the maximal grid block delta (according to neighborsDistance)
			if ( neighborsRadius < 0f ) {
				/*
				// No radius was given, take the average dimension of a block size as a radius
				int sumBlockDimSize = 0;
				for ( int d = 0; d < dim; d++ ) {
					sumBlockDimSize += gridBlockDimSize[ d ];
				}		
				neighborsRadius = (float)sumBlockDimSize / dim;
				*/
				neighborsRadius = computeDefaultRadius();
			}
			//neighborsRadius = 1024;
			System.out.println( "neigborsRadius = " + neighborsRadius );

			deltaBlock = new int[ dim ];
			for ( int d = 0; d < dim; d++ ) {
				deltaBlock[ d ] = (int)(neighborsRadius / gridBlockDimSize[ d ]) + 1;
				System.out.println( "deltaBlock[" + d + "] = " + deltaBlock[ d ] );
			}

			dataReady = true;
		}
	}
	
	@Override
	public void applyInPlace( final float[] location )
	{
		if ( dataReady == false )
			refreshGrid();

		int[] outGridDim = new int[ location.length ];
		calculateGridLocation( location, outGridDim );
		
		
		List< Integer > relevantIndices = new ArrayList< Integer >();
		List< Float > relevantWeights = new ArrayList< Float >();
		
		// Search for the neighboring points (according to the radius) in the neighboring grid blocks and the current grid block
		//int relevantIndex = 0;
		float sumW = 0;
		float doubleNeighborsRadius = neighborsRadius * neighborsRadius + EPS; // So we won't have to execute sqrt
		for ( int blockX = Math.max( 0, outGridDim[ 0 ] - deltaBlock[ 0 ] ); blockX <= Math.min( buckets.length - 1, outGridDim[ 0 ] + deltaBlock[ 0 ] ); blockX++ ) {
			for ( int blockY = Math.max( 0, outGridDim[ 1 ] - deltaBlock[ 1 ] ); blockY <= Math.min( buckets[blockX].length - 1, outGridDim[ 1 ] + deltaBlock[ 1 ] ); blockY++ ) {
				for ( int i = 0; i < buckets[blockX][blockY].size(); i++ ) {
					int pIndex = (Integer)buckets[blockX][blockY].get( i );
					float dist = 0;
					for ( int d = 0; d < location.length; d++ ) {
						float diff = p[d][pIndex] - location[d];
						dist += diff * diff;
					}
					// Found a point on an exact match, just set the location to the matched point, and return
					if ( dist < EPS ) {
						for ( int d = 0; d < location.length; ++d )
							location[ d ] = q[ d ][ pIndex ];
						return;
					}
					if ( dist <= doubleNeighborsRadius ) {
						// Add the point's data to the "list" of relevant points
						/*
						for ( int d = 0; d < location.length; d++ ) {
							relevantP[ d ][ relevantIndex ] = p[ d ][ pIndex ];
							relevantQ[ d ][ relevantIndex ] = q[ d ][ pIndex ];
						}
						relevantW[ relevantIndex ] = w[ pIndex ] * ( float )weigh( dist );
						*/
						relevantIndices.add( pIndex );
						relevantWeights.add( w[ pIndex ] * ( float )weigh( dist ) );
						sumW += dist;
						// relevantIndex++;
					}
				}
			}
		}
		
		
		/*
		// Put zero weights on all indices of the non-relevant points
		for ( int i = relevantIndex; i < relevantW.length; i++ ) {
			// Stop at the first weight that is zero (all the rest after will be 0)
			if ( relevantW[ i ] == 0 )
				break;
			relevantW[ i ] = 0;
		}
		
		// normalize the weights
		for ( int i = 0; i < relevantIndex; i++ ) {
			relevantW[ i ] = relevantW[ i ] / sumW;
		}
		*/
		/*
		if ( relevantIndex != p[0].length ) {
			System.out.print( "Seems like an error when computing location: " + location[0] + ", " + location[1] );
			System.out.println( "  relevantIndex: " + relevantIndex + ", p[0].length: " + p[0].length );
		}
		*/
		
		float[][] curP = new float[ location.length ][ relevantIndices.size() ];
		float[][] curQ = new float[ location.length ][ relevantIndices.size() ];
		float[] curW = new float[ relevantIndices.size() ];
		// normalize the weights
		/*
		for ( int i = 0; i < relevantIndex; i++ ) {
			for ( int d = 0; d < location.length; d++ ) {
				curP[ d ][ i ] = relevantP[ d ][ i ];
				curQ[ d ][ i ] = relevantQ[ d ][ i ];
			}
			//curW[ i ] = relevantW[ i ] / sumW;
			curW[ i ] = relevantW[ i ];
		}
		*/
		for ( int i = 0; i < relevantIndices.size(); i++ ) {
			int relevantIndex = relevantIndices.get( i );
			for ( int d = 0; d < location.length; d++ ) {
				curP[ d ][ i ] = p[ d ][ relevantIndex ];
				curQ[ d ][ i ] = q[ d ][ relevantIndex ];
			}
			curW[ i ] = relevantWeights.get( i );
		}
		
		try 
		{
			synchronized ( model ) {
				//model.fit( relevantP, relevantQ, relevantW );
				model.fit( curP, curQ, curW );
				model.applyInPlace( location );
			}
		}
		catch ( IllDefinedDataPointsException e ){}
		catch ( NotEnoughDataPointsException e ){}

		/*
		final float[] ww = new float[ w.length ];
		for ( int i = 0; i < w.length; ++i )
		{
			float s = 0;
			for ( int d = 0; d < location.length; ++d )
			{
				final float dx = p[ d ][ i ] - location[ d ];
				s += dx * dx;
			}
			if ( s <= 0 )
			{
				for ( int d = 0; d < location.length; ++d )
					location[ d ] = q[ d ][ i ];
				return;
			}
			ww[ i ] = w[ i ] * ( float )weigh( s );
		}
		
		try 
		{
			model.fit( p, q, ww );
			model.applyInPlace( location );
		}
		catch ( IllDefinedDataPointsException e ){}
		catch ( NotEnoughDataPointsException e ){}
		*/
	}

	
	public static List< PointMatch > filterMatches(
			final List< PointMatch > allMatches,
			final float[] boundingBox )
	{
		// The 2D boundingBox is made of 4 floats: left right top bottom
		List< PointMatch > filteredMatches = new ArrayList<PointMatch>();
		
		for ( PointMatch pm : allMatches ) {
			float[] point = pm.getP1().l;
			if ( ( point[0] >= boundingBox[0] ) && ( point[0] <= boundingBox[1] ) &&
				 ( point[1] >= boundingBox[2] ) && ( point[1] <= boundingBox[3] ) ) {
				filteredMatches.add( pm );
			}
		}
		
		return filteredMatches;
	}
	
	private List< PointMatch > filterMatches(
			final float[] boundingBox )
	{
		// The 2D boundingBox is made of 4 floats: left right top bottom
		List< PointMatch > filteredMatches = new ArrayList<PointMatch>();
		
		for ( int i = 0; i < p[0].length; i++ ) {
			if ( ( p[0][i] >= boundingBox[0] ) && ( p[0][i] <= boundingBox[1] ) &&
				 ( p[1][i] >= boundingBox[2] ) && ( p[1][i] <= boundingBox[3] ) ) {
				float[] local = { p[0][i], p[1][i] };
				float[] world = { q[0][i], q[1][i] };
				PointMatch pm = new PointMatch(
						new Point( local, local ),
						new Point( world, world ),
						w[i] );
				filteredMatches.add( pm );
			}
		}
		
		return filteredMatches;
	}

	protected List< PointMatch > filterMatchesWithHalo(
			final float[] boundingBox )
	{
		// Calcualte radius (if not calculated yet)
		if ( neighborsRadius < 0f ) {
			neighborsRadius = computeDefaultRadius();
		}
		
		float[] bboxWithHalo = new float[ boundingBox.length ];
		for ( int i = 0; i < boundingBox.length; i+= 2 ) {
			bboxWithHalo[ i ] = boundingBox[ i ] - neighborsRadius;
			bboxWithHalo[ i + 1 ] = boundingBox[ i + 1 ] + neighborsRadius;
		}
		
		return filterMatches( bboxWithHalo );
	}
	
	public RestrictedMovingLeastSquaresTransform2 boundToBoundingBox(
			final float[] boundingBox )
	{
		List< PointMatch > filteredMatches = filterMatchesWithHalo( boundingBox );
		RestrictedMovingLeastSquaresTransform2 boundedRMLT = new RestrictedMovingLeastSquaresTransform2();
		boundedRMLT.setModel( this.model );
		boundedRMLT.setAlpha( this.alpha );
		try {
			boundedRMLT.setMatches( filteredMatches );
		} catch (NotEnoughDataPointsException e) {
			// Should not happen
			throw new RuntimeException( e );
		} catch (IllDefinedDataPointsException e) {
			// Should not happen
			throw new RuntimeException( e );
		}		
		boundedRMLT.setRadius( this.neighborsRadius );

		return boundedRMLT;
	}

	
	/**
	 * Set the control points.  {@link PointMatch PointMatches} are not stored
	 * by reference but their data is copied into internal data buffers.
	 * 
	 * @param matches 
	 */
	@Override
	public void setMatches( final Collection< PointMatch > matches )
		throws NotEnoughDataPointsException, IllDefinedDataPointsException
	{
		/*
		 * fragile test for number of dimensions, we expect data to be
		 * consistent
		 */
		final int n = ( matches.size() > 0 ) ? matches.iterator().next().getP1().getL().length : 0;
		
		p = new float[ n ][ matches.size() ];
		q = new float[ n ][ matches.size() ];
		w = new float[ matches.size() ];
		
		int i = 0;
		for ( final PointMatch match : matches )
		{
			final float[] pp = match.getP1().getL();
			final float[] qq = match.getP2().getW();
			
			for ( int d = 0; d < n; ++d )
			{
				p[ d ][ i ] = pp[ d ];
				q[ d ][ i ] = qq[ d ];
			}
			w[ i ] = match.getWeight();
			++i;
		}
		if ( n > 0 )
			model.fit( p, q, w );
		else
			throw new NotEnoughDataPointsException( "No matches passed." );
	}
	
	/**
	 * <p>Set the control points passing them as arrays that are used by
	 * reference.  The leading index is dimension which usually results in a
	 * reduced object count.   E.g. four 2d points are:</p>
	 * <pre>
	 * float[][]{
	 *   {x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, x<sub>4</sub>},
	 *   {y<sub>1</sub>, y<sub>2</sub>, y<sub>3</sub>, y<sub>4</sub>} }
	 * </pre>
	 * 
	 * @param p source points
	 * @param q target points
	 * @param w weights
	 */
	final public void setMatches(
			final float[][] p,
			final float[][] q,
			final float[] w )
		throws NotEnoughDataPointsException, IllDefinedDataPointsException
	{
		this.p = p;
		this.q = q;
		this.w = w;
		
		model.fit( p, q, w );
	}

}

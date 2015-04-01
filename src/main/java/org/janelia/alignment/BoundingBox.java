package org.janelia.alignment;

import mpicbg.models.CoordinateTransform;

public class BoundingBox {

	/* Helping classes */
	public static class Point3D {
		private int x;
		private int y;
		private int z;
		
		public Point3D( int x, int y, int z ) {
			set( x, y, z );
		}

		public Point3D( int x, int y ) {
			this( x, y, -1 );
		}

		public Point3D( Point3D other ) {
			set( other.x, other.y, other.z );
		}

		public int getX() {
			return x;
		}

		public int getY() {
			return y;
		}

		public int getZ() {
			return z;
		}
		
		public void set( int x, int y, int z ) {
			this.x = x;
			this.y = y;
			this.z = z;
		}

	}
	
	public static class Dimensions {
		private int width;
		private int height;
		private int depth;
		
		public Dimensions( int width, int height, int depth ) {
			this.width = width;
			this.height = height;
			this.depth = depth;
		}
		
		public Dimensions( int width, int height ) {
			this( width, height, 1 );
		}
		
		public int getWidth() {
			return width;
		}

		public int getHeight() {
			return height;
		}

		public int getDepth() {
			return depth;
		}
	}

	
	/* Data members */
	
	private Point3D startPoint;
	private Point3D endPoint;
	private boolean initialized;
	
	/* Constructors */
	
	public BoundingBox() {
		initialized = false;
	}
	
	public BoundingBox( int leftX, int rightX, int topY, int bottomY, int frontZ, int backZ ) {
		startPoint = new Point3D( leftX, topY, frontZ );
		endPoint = new Point3D( rightX, bottomY, backZ );
		initialized = true;
	}
	
	public BoundingBox( int leftX, int rightX, int topY, int bottomY ) {
		startPoint = new Point3D( leftX, topY );
		endPoint = new Point3D( rightX, bottomY );
		initialized = true;
	}

	/* Public methods */
	
	public Point3D getStartPoint() {
		return startPoint;
	}

	public Point3D getEndPoint() {
		return endPoint;
	}

	public int getWidth() {
		return endPoint.getX() - startPoint.getX() + 1;
	}

	public int getHeight() {
		return endPoint.getY() - startPoint.getY() + 1;
	}

	public int getDepth() {
		return endPoint.getZ() - startPoint.getZ() + 1;
	}

	public void extendByWHD( int x, int width, int y, int height, int z, int depth ) {
		if ( initialized ) {
			int[] minmaxX = { startPoint.getX(), endPoint.getX() };
			int[] minmaxY = { startPoint.getY(), endPoint.getY() };
			int[] minmaxZ = { startPoint.getZ(), endPoint.getZ() };
			
			minmaxX[0] = Math.min( minmaxX[0], x );
			minmaxX[1] = Math.max( minmaxX[1], x + width );
			minmaxY[0] = Math.min( minmaxY[0], y );
			minmaxY[1] = Math.max( minmaxY[1], y + height );
			minmaxZ[0] = Math.min( minmaxZ[0], z );
			minmaxZ[1] = Math.max( minmaxZ[1], z + depth );
			
			startPoint.set( minmaxX[0], minmaxY[0], minmaxZ[0] );
			endPoint.set( minmaxX[1], minmaxY[1], minmaxZ[1] );
		} else {
			startPoint = new Point3D( x, y, z );
			endPoint = new Point3D( x + width, y + height, z + depth );
			initialized = true;
		}
	}

	public void extendByMinMax( int left, int right, int top, int bottom, int front, int back ) {
		
		if ( initialized ) {
			int[] minmaxX = { startPoint.getX(), endPoint.getX() };
			int[] minmaxY = { startPoint.getY(), endPoint.getY() };
			int[] minmaxZ = { startPoint.getZ(), endPoint.getZ() };
			
			minmaxX[0] = Math.min( minmaxX[0], left );
			minmaxX[1] = Math.max( minmaxX[1], right );
			minmaxY[0] = Math.min( minmaxY[0], top );
			minmaxY[1] = Math.max( minmaxY[1], bottom );
			minmaxZ[0] = Math.min( minmaxZ[0], front );
			minmaxZ[1] = Math.max( minmaxZ[1], back );
			
			startPoint.set( minmaxX[0], minmaxY[0], minmaxZ[0] );
			endPoint.set( minmaxX[1], minmaxY[1], minmaxZ[1] );
		} else {
			startPoint = new Point3D( left, top, front );
			endPoint = new Point3D( right, bottom, back );
			initialized = true;
		}
	}

	public void extendByMinMax( int left, int right, int top, int bottom ) {
		if ( initialized ) {
			int[] minmaxX = { startPoint.getX(), endPoint.getX() };
			int[] minmaxY = { startPoint.getY(), endPoint.getY() };
			int[] minmaxZ = { startPoint.getZ(), endPoint.getZ() };
			
			minmaxX[0] = Math.min( minmaxX[0], left );
			minmaxX[1] = Math.max( minmaxX[1], right );
			minmaxY[0] = Math.min( minmaxY[0], top );
			minmaxY[1] = Math.max( minmaxY[1], bottom );
			
			startPoint.set( minmaxX[0], minmaxY[0], minmaxZ[0] );
			endPoint.set( minmaxX[1], minmaxY[1], minmaxZ[1] );
		} else {
			startPoint = new Point3D( left, top );
			endPoint = new Point3D( right, bottom );
			initialized = true;
		}
	}

	
	public void extendByBoundingBox( BoundingBox other ) {
		if ( initialized ) {
			int[] minmaxX = { startPoint.getX(), endPoint.getX() };
			int[] minmaxY = { startPoint.getY(), endPoint.getY() };
			int[] minmaxZ = { startPoint.getZ(), endPoint.getZ() };
			
			minmaxX[0] = Math.min( minmaxX[0], other.startPoint.getX() );
			minmaxX[1] = Math.max( minmaxX[1], other.endPoint.getX() );
			minmaxY[0] = Math.min( minmaxY[0], other.startPoint.getY() );
			minmaxY[1] = Math.max( minmaxY[1], other.endPoint.getY() );
			minmaxZ[0] = Math.min( minmaxZ[0], other.startPoint.getZ() );
			minmaxZ[1] = Math.max( minmaxZ[1], other.endPoint.getZ() );
			
			startPoint.set( minmaxX[0], minmaxY[0], minmaxZ[0] );
			endPoint.set( minmaxX[1], minmaxY[1], minmaxZ[1] );
		} else {
			startPoint = new Point3D( other.startPoint );
			endPoint = new Point3D( other.endPoint );
			initialized = true;
		}
	}

	public void extendZ( int front, int back ) {
		if ( initialized ) {
			int[] minmaxZ = { startPoint.getZ(), endPoint.getZ() };
			
			if (( minmaxZ[0] == -1 ) || ( front < minmaxZ[0] ))
				minmaxZ[0] = front;
			if (( minmaxZ[1] == -1 ) || ( minmaxZ[1] < back ))
				minmaxZ[1] = back;
			
			startPoint.set( startPoint.getX(), startPoint.getY(), minmaxZ[0] );
			endPoint.set( endPoint.getX(), endPoint.getY(), minmaxZ[1] );
		} else {
			throw new UnsupportedOperationException();
		}
	}

	public boolean isInitialized() {
		return initialized;
	}
	
	public String toString() {
		return startPoint.getX() + " " + endPoint.getX() + " " +
				startPoint.getY() + " " + endPoint.getY() + " " +
				startPoint.getZ() + " " + endPoint.getZ() ;
	}

	public float[] to2DFloatArray() {
		float[] arr = {
			startPoint.getX(),
			endPoint.getX(),
			startPoint.getY(),
			endPoint.getY()
		};
		return arr;
	}
	
	public BoundingBox apply2DAffineTransformation( CoordinateTransform model ) {
		if ( !initialized )
			throw new RuntimeException( "Cannot apply transformation to a non intialized bounding box" );
		
		// Apply the transformation to all boundary points
		float[][] corners = new float[4][];
		corners[0] = new float[] { startPoint.getX(), startPoint.getY() }; // top-left
		corners[1] = new float[] { endPoint.getX(), startPoint.getY() }; // top-right
		corners[2] = new float[] { startPoint.getX(), endPoint.getY() }; // bottom-left
		corners[3] = new float[] { endPoint.getX(), endPoint.getY() }; // bottom-right
		for ( int c = 0; c < corners.length; c++ )
			model.applyInPlace( corners[ c ] );
		
		// Find the new bounding box
		int[] minmaxX = { Math.round( corners[0][0] ), Math.round( corners[0][0] ) };
		int[] minmaxY = { Math.round( corners[0][1] ), Math.round( corners[0][1] ) };
		for ( int c = 1; c < corners.length; c++ ) {
			minmaxX[0] = Math.min( minmaxX[0], Math.round( corners[ c ][ 0 ] ) );
			minmaxX[1] = Math.max( minmaxX[1], Math.round( corners[ c ][ 0 ] ) );
			minmaxY[0] = Math.min( minmaxY[0], Math.round( corners[ c ][ 1 ] ) );
			minmaxY[1] = Math.max( minmaxY[1], Math.round( corners[ c ][ 1 ] ) );
		}

		return new BoundingBox( minmaxX[0], minmaxX[1], minmaxY[0], minmaxY[1], startPoint.getZ(), endPoint.getZ() );
	}
	
	/**
	 * Returns true if there is intersection between the bboxes or a full containment
	 * @param other
	 * @return
	 */
	public boolean overlap( final BoundingBox other ) {
		if ( ( startPoint.getX() < other.endPoint.getX() ) && ( endPoint.getX() > other.startPoint.getX() ) &&
			 ( startPoint.getY() < other.endPoint.getY() ) && ( endPoint.getY() > other.startPoint.getY() ) )
			return true;
		return false;
	}
}

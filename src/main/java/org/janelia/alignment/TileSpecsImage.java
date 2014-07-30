package org.janelia.alignment;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import mpicbg.models.AffineModel2D;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.CoordinateTransformMesh;
import mpicbg.models.PointMatch;
import mpicbg.models.TransformMesh;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks.ImageProcessorWithMasks;
import mpicbg.util.Util;

/**
 * A 3D image representation that can be created by a collection of tile specs
 * (each tile can be a part of a section in the 3D space)
 */
public class TileSpecsImage {

	private static final int DEFAULT_TRIANGLE_SIZE = 64;
	
	/* Data members */
	
	private List< TileSpec > tileSpecs;
	private BoundingBox boundingBox;
	private double triangleSize;
	// The number of threads used to perfrom the rendering and bounding box computation
	private int threadsNum;
	private boolean parsed;
	
	
	/* C'tors */
	
	/**
	 * An image based on the given tile specs
	 */
	public TileSpecsImage( List< TileSpec > tileSpecs ) {
		this( tileSpecs, DEFAULT_TRIANGLE_SIZE );
	}

	public TileSpecsImage( List< TileSpec > tileSpecs, int triangleSize ) {
		this.tileSpecs = tileSpecs;
		this.triangleSize = triangleSize;
		initialize();
	}

	/**
	 * An image based on the given tile specs
	 */
	public TileSpecsImage( TileSpec[] tileSpecsArr ) {
		this( tileSpecsArr, DEFAULT_TRIANGLE_SIZE );
	}

	public TileSpecsImage( TileSpec[] tileSpecsArr, int triangleSize ) {
		this.tileSpecs = Arrays.asList( tileSpecsArr );
		this.triangleSize = triangleSize;
		initialize();
	}

	
	/* Public methods */

	public void setThreadsNum( int threadsNum ) {
		this.threadsNum = threadsNum;
	}
	
	public void renderAndSave( String outFile, int layer, int mipmapLevel ) {
		renderAndSave( outFile, layer, mipmapLevel, 1.0f );
	}

	public void renderAndSave( String outFile, int layer, int mipmapLevel, float scale ) {
		ColorProcessor cp = render( layer, mipmapLevel, scale );
		IJ.save( new ImagePlus( "Layer " + layer, cp ), outFile );
	}

	public ColorProcessor render( int layer, int mipmapLevel ) {
		return render( layer, mipmapLevel, 1.0f );
	}
	
	public ColorProcessor render( int layer, int mipmapLevel, float scale ) {
		
		// Get image width and height
		parseTileSpecs();
		
		/* create a target */
		ByteProcessor tp = new ByteProcessor( (int) (boundingBox.getWidth() * scale),
				(int) (boundingBox.getHeight() * scale) );
		ColorProcessor cp = null;
		
		// Create an offset according to the bounding box
		final int offsetX = 0; //boundingBox.getStartPoint().getX();
		final int offsetY = 0; //boundingBox.getStartPoint().getY();
		
		final ExecutorService threadPool = Executors.newFixedThreadPool( threadsNum );
		
		for ( TileSpec ts : tileSpecs ) {

			if ( ts.layer != layer )
				continue;
			
			ImageAndMask tsMipmapEntry = null;
			ImageProcessor tsIp = null;
			ImageProcessor tsIpMipmap = null;

			
			TreeMap< String, ImageAndMask > tsMipmapLevels = ts.getMipmapLevels();
			
			String key = tsMipmapLevels.floorKey( String.valueOf( mipmapLevel ) );
			if ( key == null )
				key = tsMipmapLevels.firstKey();
			
			/* load image TODO use Bioformats for strange formats */
			tsMipmapEntry = tsMipmapLevels.get( key );
			final String imgUrl = tsMipmapEntry.imageUrl;
			final ImagePlus imp = Utils.openImagePlusUrl( imgUrl );
			if ( imp == null )
			{
				System.err.println( "Failed to load image '" + imgUrl + "'." );
				continue;
			}
			tsIp = imp.getProcessor();
			final int currentMipmapLevel = Integer.parseInt( key );
			if ( currentMipmapLevel >= mipmapLevel )
			{
				mipmapLevel = currentMipmapLevel;
				tsIpMipmap = tsIp;
			}
			else
				tsIpMipmap = Downsampler.downsampleImageProcessor( tsIp, mipmapLevel - currentMipmapLevel );

			
			/* open mask */
			final ByteProcessor bpMaskSource;
			final ByteProcessor bpMaskTarget;
			final String maskUrl = tsMipmapEntry.maskUrl;
			if ( maskUrl != null )
			{
				final ImagePlus impMask = Utils.openImagePlusUrl( maskUrl );
				if ( impMask == null )
				{
					System.err.println( "Failed to load mask '" + maskUrl + "'." );
					bpMaskSource = null;
					bpMaskTarget = null;
				}
				else
				{
					/* create according mipmap level */
					bpMaskSource = Downsampler.downsampleByteProcessor( impMask.getProcessor().convertToByteProcessor(), mipmapLevel );
					bpMaskTarget = new ByteProcessor( tp.getWidth(), tp.getHeight() );
				}
			}
			else
			{
				bpMaskSource = null;
				bpMaskTarget = null;
			}
			
			
			/* attach mipmap transformation */
			final CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
			final AffineModel2D scaleTransform = new AffineModel2D();
			scaleTransform.set( ( float )scale, 0, 0, ( float )scale, -( float )( offsetX * scale ), -( float )( offsetY * scale ) );

			ctl.add( scaleTransform );

			final CoordinateTransformList< CoordinateTransform > ctlMipmap = new CoordinateTransformList< CoordinateTransform >();
			ctlMipmap.add( Utils.createScaleLevelTransform( mipmapLevel ) );
			ctlMipmap.add( ctl );
			
			/* create mesh */
			final CoordinateTransformMesh mesh = new CoordinateTransformMesh( ctlMipmap,  ( int )( boundingBox.getWidth() / triangleSize + 0.5 ), tsIpMipmap.getWidth(), tsIpMipmap.getHeight() );
			
			final ImageProcessorWithMasks source = new ImageProcessorWithMasks( tsIpMipmap, bpMaskSource, null );
			final ImageProcessorWithMasks target = new ImageProcessorWithMasks( tp, bpMaskTarget, null );
			final TransformMeshMappingWithMasks< TransformMesh > mapping = new TransformMeshMappingWithMasks< TransformMesh >( mesh );
			mapping.mapInterpolated( source, target, threadsNum );
			
			/* convert to 24bit RGB */
			tp.setMinAndMax( ts.minIntensity, ts.maxIntensity );
			cp = tp.convertToColorProcessor();

			final int[] cpPixels = ( int[] )cp.getPixels();
			final byte[] alphaPixels;
			
			
			/* set alpha channel */
			if ( bpMaskTarget != null )
				alphaPixels = ( byte[] )bpMaskTarget.getPixels();
			else
				alphaPixels = ( byte[] )target.outside.getPixels();


			if ( threadsNum == 1 )
			{
				for ( int i = 0; i < cpPixels.length; ++i )
					cpPixels[ i ] &= 0x00ffffff | ( alphaPixels[ i ] << 24 );
			}
			else
			{
				final int pixelsPerThread = cpPixels.length / threadsNum;
				final List< Future< ? > > futures = new ArrayList< Future< ? >>();
	
				for ( int t = 0; t < threadsNum; t++ ) {
					final int threadIndex = t;
					final Future< ? > future = threadPool.submit( new Runnable() {
	
						@Override
						public void run() {
							int startIndex = threadIndex * pixelsPerThread;
							int endIndex = ( threadIndex + 1 ) * pixelsPerThread;
							if ( threadIndex == threadsNum - 1 )
								endIndex = cpPixels.length;
							for ( int i = startIndex; i < endIndex; ++i )
								cpPixels[ i ] &= 0x00ffffff | ( alphaPixels[ i ] << 24 );
							
						}
					});
					futures.add( future );
				}
				
				try {
					for ( Future< ? > future : futures ) {
						future.get();
					}
				} catch ( InterruptedException e ) {
					e.printStackTrace();
					throw new RuntimeException( e );
				} catch ( ExecutionException e ) {
					e.printStackTrace();
					throw new RuntimeException( e );
				}
			}
		}
		
		threadPool.shutdown();
		
		return cp;
	}

	
	public BoundingBox getBoundingBox() {
		if ( boundingBox == null ) {
			parseTileSpecs();
		}
		return boundingBox;
	}

	public BoundingBox getBoundingBox( boolean froceRecomputation ) {
		parseTileSpecs(froceRecomputation);
		return boundingBox;
	}

	public void saveTileSpecs( String fileName ) {
		try {
			Writer writer = new FileWriter( fileName );
	        Gson gson = new GsonBuilder().setPrettyPrinting().create();
	        gson.toJson( tileSpecs, writer );
	        writer.close();
	    }
		catch ( final IOException e )
		{
			System.err.println( "Error writing JSON file: " + fileName );
			e.printStackTrace( System.err );
		}

	}
	
	/**
	 * Creates a TileSpecsImage out of a given set of tile spec json files.
	 * 
	 * @param fileNames
	 * @return
	 */
	public static TileSpecsImage createImageFromFiles( List< String > fileNames ) {
		
		// Parse all files
		TileSpec[] allTileSpecs = TileSpecUtils.readTileSpecFile( fileNames.toArray( new String[0] ) );
		
		List< TileSpec > tileSpecs = Arrays.asList( allTileSpecs );
		
		if ( tileSpecs.size() == 0 ) 
			throw new RuntimeException( "No tile spec found, aborting!" );
		
		TileSpecsImage image = new TileSpecsImage( tileSpecs );
		return image;
	}

	/**
	 * Creates a TileSpecsImage out of a given tile spec json file.
	 * 
	 * @param fileNames
	 * @return
	 */
	public static TileSpecsImage createImageFromFile( String fileName ) {
		List< String > fileNames = new ArrayList< String >();
		fileNames.add( fileName );
		return createImageFromFiles( fileNames );
	}

	/* Helping methods */

	private void initialize() {
		boundingBox = null;
		threadsNum = Runtime.getRuntime().availableProcessors();
		parsed = false;
	}
	

	/**
	 * Computes the dimensions, the start point of the 3D image, and its bounding box
	 */
	private void parseTileSpecs() {
		parseTileSpecs( 0, false );
	}

	private void parseTileSpecs( boolean recomputeBoundingBox ) {
		parseTileSpecs( 0, recomputeBoundingBox );
	}

	/**
	 * Computes the dimensions, the start point of the 3D image, and its bounding box
	 */
	private void parseTileSpecs( int mipmapLevel, boolean recomputeBoundingBox ) {
		
		if ( parsed == false )
		{
			synchronized( this )
			{
				if ( parsed == false )
				{
					// Iterate through the tiles, find the width and height (after applying the transformations),
					// and the depth of the image
					int[] minmaxZ = { Integer.MAX_VALUE, Integer.MIN_VALUE };
					
					// Create an uninitialized bounding box
					boundingBox = new BoundingBox();
					
					/* Iterate through the tile specs */
					System.out.println( "Parsing all tilespecs." );
					for ( TileSpec ts : tileSpecs ) {
						// Update the Z value
						if ( ts.layer != -1 ) {
							minmaxZ[0] = Math.min( minmaxZ[0], ts.layer );
							minmaxZ[1] = Math.max( minmaxZ[1], ts.layer );
						}
						
						// Get the bounding box of the tilespec
						BoundingBox bbox;
						if ( ( ts.bbox == null ) || ( recomputeBoundingBox ) ) {
							bbox = getTileSpecBoundingBox( ts, mipmapLevel );
							ts.bbox = bbox.to2DFloatArray();
						} else {
							bbox = new BoundingBox( 
									(int) ts.bbox[0],
									(int) ts.bbox[1],
									(int) ts.bbox[2],
									(int) ts.bbox[3]
									);
						}
						boundingBox.extendByBoundingBox( bbox );
						boundingBox.extendZ( minmaxZ[0], minmaxZ[1] );
					}
					
					if ( !boundingBox.isInitialized() )
						throw new RuntimeException( "Error: failed to parse tile specs" );

					System.out.println( "Parsing all tilespecs - Done." );
					System.out.println( "all tilespecs bounding box: " + boundingBox );
					parsed = true;
				}
			}
		}


	}
	
	private BoundingBox getTileSpecBoundingBox( TileSpec ts, int mipmapLevel ) {
		// Read the image original height and width
		/* obtain available mipmap urls as a sorted map */
		final TreeMap< String, ImageAndMask > mipmapLevels = ts.getMipmapLevels();

		ImageAndMask mipmapEntry = mipmapLevels.get( String.valueOf( mipmapLevel ) );
		if ( mipmapEntry == null )
			throw new RuntimeException( "Failed to load mipmap level " + mipmapLevel + " from tilespec." );
		
		/* load image TODO use Bioformats for strange formats */
		final String imgUrl = mipmapEntry.imageUrl;
		final ImagePlus imp = Utils.openImagePlusUrl( imgUrl );
		if ( imp == null )
			throw new RuntimeException( "Failed to load image '" + imgUrl + "'." );
		final ImageProcessor ip = imp.getProcessor();
		final int origWidth = imp.getWidth();
		//final int origHeight = imp.getHeight();
		
		// Apply the transformations to the image
		final CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
		/* create mesh */
		final CoordinateTransformMesh mesh = new CoordinateTransformMesh( ctl,  ( int )( origWidth / triangleSize + 0.5 ), ip.getWidth(), ip.getHeight() );
		final BoundingBox boundingBox = findBoundingBox( mesh, ip, threadsNum );
		System.out.println( " tilespec bounding box is: " + boundingBox );

		// Create the corresponding bounding box, and return it
		return boundingBox;
		
	}
	

	final static private BoundingBox findBoundingBox(
			final TransformMesh transform,
			final ImageProcessor source )
	{
		
		return findBoundingBox( transform, source, Runtime.getRuntime().availableProcessors() );
	}

	
	final static private BoundingBox findBoundingBox(
			final TransformMesh transform,
			final ImageProcessor source,
			final int numThreads )
	{
		// Create an uninitialized bounding box
		final BoundingBox boundingBox = new BoundingBox();
		
		if ( numThreads == 1 )
		{

			/* no overhead for thread creation */
			final Set< AffineModel2D > s = transform.getAV().keySet();
			//System.out.println( "BoundingBox transform keySet size: " + s.size() );
			for ( final AffineModel2D ai : s )
				findBoundingBox( transform, ai, source, boundingBox );

		}
		else
		{
			final AffineModel2D[] s = transform.getAV().keySet().toArray( new AffineModel2D[ 0 ] );
			final int modelsPerThread = s.length / numThreads;
			final ExecutorService threadPool = Executors.newFixedThreadPool( numThreads );
			
			final BoundingBox[] threadBoundingBoxes = new BoundingBox[ numThreads ];
			final Future< ? >[] futures = new Future< ? >[ numThreads ];
			
			for ( int t = 0; t < numThreads; t++ ) {
				final int threadIndex = t;
				threadBoundingBoxes[ t ] = new BoundingBox();
				futures[ t ] = threadPool.submit( new Runnable() {

					@Override
					public void run() {
						int startIndex = threadIndex * modelsPerThread;
						int endIndex = ( threadIndex + 1 ) * modelsPerThread;
						if ( threadIndex == numThreads - 1 )
							endIndex = s.length;
						for ( int i = startIndex; i < endIndex; ++i )
							findBoundingBox( transform, s[ i ], source, threadBoundingBoxes[ threadIndex ] );						
					}
				});
			}
			
			try {
				for ( int t = 0; t < numThreads; t++ ) {
					futures[ t ].get();
					boundingBox.extendByBoundingBox( threadBoundingBoxes[ t ]);
				}
			} catch ( InterruptedException e ) {
				e.printStackTrace();
				throw new RuntimeException( e );
			} catch ( ExecutionException e ) {
				e.printStackTrace();
				throw new RuntimeException( e );
			}

			threadPool.shutdown();
		}

		return boundingBox;
	}


	/**
	 * 
	 * @param pm PointMatches
	 * @param min x = min[0], y = min[1]
	 * @param max x = max[0], y = max[1]
	 */
	final static private void calculateTriangleBoundingBox(
			final ArrayList< PointMatch > pm,
			final float[] min,
			final float[] max )
	{
		//final float[] first = pm.get( 0 ).getP2().getW();
		final float[] first = pm.get( 0 ).getP2().getL();
		min[ 0 ] = first[ 0 ];
		min[ 1 ] = first[ 1 ];
		max[ 0 ] = first[ 0 ];
		max[ 1 ] = first[ 1 ];
		
		for ( final PointMatch p : pm )
		{
			//final float[] t = p.getP2().getW();
			final float[] t = p.getP2().getL();
			if ( t[ 0 ] < min[ 0 ] ) min[ 0 ] = t[ 0 ];
			else if ( t[ 0 ] > max[ 0 ] ) max[ 0 ] = t[ 0 ];
			if ( t[ 1 ] < min[ 1 ] ) min[ 1 ] = t[ 1 ];
			else if ( t[ 1 ] > max[ 1 ] ) max[ 1 ] = t[ 1 ];
		}
	}

	

	final static private void findBoundingBox(
			final TransformMesh m, 
			final AffineModel2D ai,
			final ImageProcessor source,
			final BoundingBox boundingBox )
	{
		final ArrayList< PointMatch > pm = m.getAV().get( ai );
		final float[] min = new float[ 2 ];
		final float[] max = new float[ 2 ];
		int[] interpolatedMin = null;
		int[] interpolatedMax = null;
		calculateTriangleBoundingBox( pm, min, max );

		final int minX;
		final int minY;
		final int maxX;
		final int maxY;
		minX = Util.roundPos( min[ 0 ] );
		minY = Util.roundPos( min[ 1 ] );
		maxX = Util.roundPos( max[ 0 ] );
		maxY = Util.roundPos( max[ 1 ] );
		
		//final float[] a = pm.get( 0 ).getP2().getW();
		final float[] a = pm.get( 0 ).getP2().getL();
		final float ax = a[ 0 ];
		final float ay = a[ 1 ];
		//final float[] b = pm.get( 1 ).getP2().getW();
		final float[] b = pm.get( 1 ).getP2().getL();
		final float bx = b[ 0 ];
		final float by = b[ 1 ];
		//final float[] c = pm.get( 2 ).getP2().getW();
		final float[] c = pm.get( 2 ).getP2().getL();
		final float cx = c[ 0 ];
		final float cy = c[ 1 ];
		final float[] t = new float[ 2 ];
		for ( int y = minY; y <= maxY; ++y )
		{
			for ( int x = minX; x <= maxX; ++x )
			{
				if ( isInTriangle( ax, ay, bx, by, cx, cy, x, y ) )
				{
					t[ 0 ] = x;
					t[ 1 ] = y;
					try
					{
						ai.applyInPlace( t );
					}
					catch ( Exception e )
					{
						//e.printStackTrace( System.err );
						continue;
					}

					// Update the coordinates of the bounding box of this triangle
					int roundT0 = Util.roundPos( t[0] );
					int roundT1 = Util.roundPos( t[1] );
					if ( interpolatedMin == null )
					{
						interpolatedMin = new int[] { roundT0, roundT1 };
						interpolatedMax = new int[] { roundT0, roundT1 };
					}
					else
					{
						interpolatedMin[0] = Math.min( interpolatedMin[0], roundT0 );
						interpolatedMin[1] = Math.min( interpolatedMin[1], roundT1 );
						interpolatedMax[0] = Math.max( interpolatedMax[0], roundT0 );
						interpolatedMax[1] = Math.max( interpolatedMax[1], roundT1 );
					}
				}
			}
		}
		if ( interpolatedMin != null)
		{
			boundingBox.extendByMinMax( 
					interpolatedMin[0], interpolatedMax[0],
					interpolatedMin[1], interpolatedMax[1]
					);
		}
	}

	
	/**
	 * Checks if a location is inside a given triangle.
	 * 
	 * @param pm
	 * @param t
	 * @return
	 */
	final static private boolean isInTriangle(
			final float ax,
			final float ay,
			final float bx,
			final float by,
			final float cx,
			final float cy,
			final float tx,
			final float ty )
	{
		final boolean d;
		{
			final float x1 = bx - ax;
			final float y1 = by - ay;
			final float x2 = tx - ax;
			final float y2 = ty - ay;
			d = x1 * y2 - y1 * x2 < 0;
		}
		{
			final float x1 = cx - bx;
			final float y1 = cy - by;
			final float x2 = tx - bx;
			final float y2 = ty - by;
			if ( d ^ x1 * y2 - y1 * x2 < 0 ) return false;
		}
		{
			final float x1 = ax - cx;
			final float y1 = ay - cy;
			final float x2 = tx - cx;
			final float y2 = ty - cy;
			if ( d ^ x1 * y2 - y1 * x2 < 0 ) return false;
		}
		return true;
	}

	
	/**
	 * Main function for short tests of this class
	 * 
	 * @param args
	 */
	public static void main( String[] args ) {
		
		String[] tsFiles = {
				  "file:///Users/adisuis/Fiji/FijiBento/scripts/alyssa_3d/my_Sec001.json",
				  "file:///Users/adisuis/Fiji/FijiBento/scripts/alyssa_3d/my_Sec002.json"
		};
		
		TileSpecsImage[] tsImgs = new TileSpecsImage[ tsFiles.length ];
		
		for ( int i = 0; i < tsFiles.length; i++ ) {
			System.out.println( "parsing " + tsFiles[ i ] );
			tsImgs[ i ] = TileSpecsImage.createImageFromFile( tsFiles[ i ] );
		}

		for ( int i = 0; i < tsFiles.length; i++ ) {
			System.out.println( "Fetching bounding box of " + tsFiles[ i ] );
			BoundingBox bbox = tsImgs[ i ].getBoundingBox();
			System.out.println( bbox );

			String outFile = tsFiles[ i ].replace( "file://", "" ).replaceAll( ".json", ".png" );
			tsImgs[ i ].renderAndSave( outFile, bbox.getStartPoint().getZ(), 0, 0.25f );
		}

		/*
		String[] layersTsFiles = {
				"file:///Users/adisuis/Fiji/FijiBento/scripts/alyssa_3d/Sec001.json",
				"file:///Users/adisuis/Fiji/FijiBento/scripts/alyssa_3d/Sec002.json"
		};

		System.out.println( "Creating a multi sections image" );
		TileSpecsImage layersTsImg = TileSpecsImage.createImageFromFiles( Arrays.asList( layersTsFiles ) );

		System.out.println( "Fetching bounding box of multi sections image" );
		System.out.println( layersTsImg.getBoundingBox() );
		*/

		/*
		String[] tsFiles = {
				"file:///Users/adisuis/Fiji/FijiBento/scripts/alyssa/Sec002.json",
				"file:///Users/adisuis/Fiji/FijiBento/scripts/alyssa/Sec002_optimized_montage.json"
		};
		
		TileSpecsImage[] tsImgs = new TileSpecsImage[ tsFiles.length ];
		
		for ( int i = 0; i < tsFiles.length; i++ ) {
			System.out.println( "parsing " + tsFiles[ i ] );
			tsImgs[ i ] = TileSpecsImage.createImageFromFile( tsFiles[ i ] );
		}

		for ( int i = 0; i < tsFiles.length; i++ ) {
			System.out.println( "Fetching bounding box of " + tsFiles[ i ] );
			System.out.println( tsImgs[ i ].getBoundingBox() );
		}
		*/
	}
}

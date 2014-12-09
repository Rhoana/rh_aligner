package org.janelia.alignment;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.TreeMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import mpicbg.models.AffineModel2D;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.CoordinateTransformMesh;
import mpicbg.models.TransformMesh;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks;
import mpicbg.trakem2.transform.TranslationModel2D;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks.ImageProcessorWithMasks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

public class RenderTiles {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--url", description = "URL to JSON tile spec", required = true )
        public String url;

        @Parameter( names = "--targetDir", description = "Directory to the output images", required = true )
        public String targetDir;
        
        @Parameter( names = "--res", description = " Mesh resolution, specified by the desired size of a triangle in pixels", required = false )
        public int res = 64;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
        @Parameter( names = "--scale", description = "scale factor applied to the target image (overrides --mipmap_level)", required = false )
        public double scale = -Double.NaN;
        
        @Parameter( names = "--quality", description = "JPEG quality float [0, 1]", required = false )
        public float quality = 0.85f;

        @Parameter( names = "--tileSize", description = "The size of a tile (one side of the square)", required = false )
        public int tileSize = 512;

	}

	
	private static class SingleTileSpecRender {
		
		public SingleTileSpecRender( final TileSpec ts, final int threadsNum ) {
			this( ts, threadsNum, DEFAULT_TRIANGLE_SIZE );
		}

		public SingleTileSpecRender( final TileSpec ts, final int threadsNum, final int triangleSize ) {
			this.ts = ts;
			this.threadsNum = threadsNum;
			this.bbox = null;
			this.renderedBp = null;
			this.triangleSize = triangleSize;
		}

		/**
		 * Fetch the current tilespec bounding box (doesn't have to start at (0,0))
		 * @return
		 */
		public BoundingBox getBoundingBox() {
			if ( bbox == null ) {
				// check if the tilespec already has a bounding box
				if ( ts.bbox == null ) {
					throw new RuntimeException( "Could not find a bounding box for a tilespec" );
				}
				
				bbox = new BoundingBox(
						(int) ts.bbox[0],
						(int) ts.bbox[1],
						(int) ts.bbox[2],
						(int) ts.bbox[3],
						ts.layer,
						ts.layer
					);
			}
			return bbox;
		}

		/**
		 * Get normalization transform that translates the tile to (0,0), or null if none is needed
		 */
		private TranslationModel2D getNormalizationTransform() {
			BoundingBox currentBBox = getBoundingBox();
			int minX = currentBBox.getStartPoint().getX();
			int minY = currentBBox.getStartPoint().getY();
			TranslationModel2D normalizationTransform = null;
			if (( minX != 0 ) || ( minY != 0 )) {
				// Add transformation (translation)
				normalizationTransform = new TranslationModel2D();
				normalizationTransform.init( -minX + ".0 " + -minY + ".0" );
			}
			return normalizationTransform;
		}

		public ByteProcessor render( ) {
			return render( 1.0f );
		}

		public ByteProcessor render( final float scale ) {
			if ( renderedBp == null ) {
				
				final BoundingBox currentBBox = getBoundingBox();
				
				TranslationModel2D normalizationTransform = getNormalizationTransform();
				final int tileWidth = currentBBox.getWidth();
				final int tileHeight = currentBBox.getHeight();
				
				final int mipmapLevel = 0;
				
				/* create a target */
				renderedBp = new ByteProcessor( (int) ( tileWidth * scale ),
						(int) ( tileHeight * scale ) );
				
				// Create an offset according to the bounding box
				final int offsetX = 0; //boundingBox.getStartPoint().getX();
				final int offsetY = 0; //boundingBox.getStartPoint().getY();
				
				
				ImageAndMask tsMipmapEntry = null;
				ImageProcessor tsIp = null;	
				
				TreeMap< String, ImageAndMask > tsMipmapLevels = ts.getMipmapLevels();
				
				String key = tsMipmapLevels.floorKey( String.valueOf( mipmapLevel ) );
				if ( key == null )
					key = tsMipmapLevels.firstKey();
				
				/* load image TODO use Bioformats for strange formats */
				tsMipmapEntry = tsMipmapLevels.get( key );
				final String imgUrl = tsMipmapEntry.imageUrl;
				System.out.println( "Rendering tile: " + imgUrl );
				final ImagePlus imp = Utils.openImagePlusUrl( imgUrl );
				if ( imp == null )
				{
					throw new RuntimeException( "Failed to load image '" + imgUrl + "'." );
				}
				tsIp = imp.getProcessor();
	
				
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
						bpMaskTarget = new ByteProcessor( renderedBp.getWidth(), renderedBp.getHeight() );
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
				if ( normalizationTransform != null )
					ctl.add( normalizationTransform );
	
				final CoordinateTransformList< CoordinateTransform > ctlMipmap = new CoordinateTransformList< CoordinateTransform >();
				ctlMipmap.add( Utils.createScaleLevelTransform( mipmapLevel ) );
				ctlMipmap.add( ctl );
				
				/* create mesh */
				final CoordinateTransformMesh mesh = new CoordinateTransformMesh( 
						ctlMipmap, 
						( int )( tileWidth / triangleSize + 0.5 ), 
						tsIp.getWidth(), 
						tsIp.getHeight(), 
						threadsNum );
				
				final ImageProcessorWithMasks source = new ImageProcessorWithMasks( tsIp, bpMaskSource, null );
				final ImageProcessorWithMasks target = new ImageProcessorWithMasks( renderedBp, bpMaskTarget, null );
				final TransformMeshMappingWithMasks< TransformMesh > mapping = new TransformMeshMappingWithMasks< TransformMesh >( mesh );
				mapping.mapInterpolated( source, target, threadsNum );
				
				/* convert to 24bit RGB */
				renderedBp.setMinAndMax( ts.minIntensity, ts.maxIntensity );
	
			}
			return renderedBp;
		}

		
		private TileSpec ts;
		private int threadsNum;
		private BoundingBox bbox;
		private ByteProcessor renderedBp;
		private int triangleSize;
		private static final int DEFAULT_TRIANGLE_SIZE = 64;

	}
	
	
	private RenderTiles() {}
	
	
	final static Params parseParams( final String[] args )
	{
		final Params params = new Params();
		try
        {
			final JCommander jc = new JCommander( params, args );
        	if ( params.help )
            {
        		jc.usage();
                return null;
            }
        }
        catch ( final Exception e )
        {
        	e.printStackTrace();
            final JCommander jc = new JCommander( params );
        	jc.setProgramName( "java [-options] -cp render.jar + " + Render.class.getCanonicalName() );
        	jc.usage(); 
        	return null;
        }
		
		/* process params */
		if ( Double.isNaN( params.scale ) )
			params.scale = 1.0;
		
		return params;
	}	
	
	private static void saveAsTiles( 
			final SingleTileSpecRender[] origTiles,
			final BoundingBox entireImageBBox,
			final int tileSize,
			final String outputDir )
	{
		BufferedImage targetImage = new BufferedImage( tileSize, tileSize, BufferedImage.TYPE_BYTE_GRAY );
		ByteProcessor targetBp = new ByteProcessor( targetImage );
		WritableRaster origRaster = targetImage.getRaster();
		Object origData = origRaster.getDataElements( 0, 0, tileSize, tileSize, null );
		WritableRaster raster = targetImage.getRaster();
		
		
		for ( int row = 0; row < entireImageBBox.getHeight(); row += tileSize )
		{
			int tileMaxRow = Math.min( row + tileSize, entireImageBBox.getHeight() );
			for ( int col = 0; col < entireImageBBox.getWidth(); col += tileSize )
			{
				int tileMaxCol = Math.min( col + tileSize, entireImageBBox.getWidth() );

				// Set each pixel of the output tile
				for ( int pixelRow = row; pixelRow < tileMaxRow; pixelRow++ ) {
					for ( int pixelCol = col; pixelCol < tileMaxCol; pixelCol++ ) {
						int sumPixelValue = 0;
						int overlapPixelCount = 0;
						
						// Iterate over all tiles that this pixel is part of, and the average of
						// this pixel's values will be used as the output value
						// Find the original tiles that are needed for the output tile
						for ( int origTileIdx = 0; origTileIdx < origTiles.length; origTileIdx++ ) {
							BoundingBox origTileBBox = origTiles[ origTileIdx ].getBoundingBox();
							if ( origTileBBox.containsPoint( pixelCol, pixelRow ) ) {
								ByteProcessor origTileBp = origTiles[ origTileIdx ].render();
								int translatedX = pixelCol - origTileBBox.getStartPoint().getX();
								int translatedY = pixelRow - origTileBBox.getStartPoint().getY();
								if ( origTileBp.get( translatedX, translatedY ) != 0 ) {
									sumPixelValue += origTileBp.get( translatedX, translatedY );
									overlapPixelCount++;
								}
							}
						}
						
						if ( overlapPixelCount != 0 ) {
							targetBp.set(
									pixelCol - col,
									pixelRow - row,
									(sumPixelValue / overlapPixelCount) );
						}
					}
					
				}
												
				// Save the image to disk
				String outFile = outputDir + File.separatorChar + "tile_" + (row / tileSize) + "_" + (col / tileSize) + ".jpg";
				Utils.saveImage( targetImage, outFile, "jpg" );
				
				// Clear the output image
				raster.setDataElements( 0, 0, tileSize, tileSize, origData );
			}			
		}
	}

	private static void saveAsTiles( 
			final SingleTileSpecRender[] origTiles,
			final BoundingBox entireImageBBox,
			final int tileSize,
			final String outputDir,
			final int threadsNum )
	{
		// Divide the rows between the threads
		final int rowsPerThread = ( entireImageBBox.getHeight() / tileSize ) / threadsNum;
		
		// Initialize threads
		final ExecutorService exec = Executors.newFixedThreadPool( threadsNum );
		final ArrayList< Future< ? > > tasks = new ArrayList< Future< ? > >();

		for ( int t = 0; t < threadsNum; t++ ) {
			final int fromRow = t * rowsPerThread * tileSize;
			final int lastRow;
			if ( t == threadsNum - 1 ) // lastThread
				lastRow = entireImageBBox.getHeight();
			else
				lastRow = fromRow + rowsPerThread * tileSize;
			
			tasks.add( exec.submit( new Runnable() {
				
				@Override
				public void run() {
					BufferedImage targetImage = new BufferedImage( tileSize, tileSize, BufferedImage.TYPE_BYTE_GRAY );
					ByteProcessor targetBp = new ByteProcessor( targetImage );
					WritableRaster origRaster = targetImage.getRaster();
					Object origData = origRaster.getDataElements( 0, 0, tileSize, tileSize, null );
					WritableRaster raster = targetImage.getRaster();

					for ( int row = fromRow; row < lastRow; row += tileSize )
					{
						int tileMaxRow = Math.min( row + tileSize, entireImageBBox.getHeight() );
						for ( int col = 0; col < entireImageBBox.getWidth(); col += tileSize )
						{
							int tileMaxCol = Math.min( col + tileSize, entireImageBBox.getWidth() );

							// Set each pixel of the output tile
							for ( int pixelRow = row; pixelRow < tileMaxRow; pixelRow++ ) {
								for ( int pixelCol = col; pixelCol < tileMaxCol; pixelCol++ ) {
									int sumPixelValue = 0;
									int overlapPixelCount = 0;
									
									// Iterate over all tiles that this pixel is part of, and the average of
									// this pixel's values will be used as the output value
									// Find the original tiles that are needed for the output tile
									for ( int origTileIdx = 0; origTileIdx < origTiles.length; origTileIdx++ ) {
										BoundingBox origTileBBox = origTiles[ origTileIdx ].getBoundingBox();
										if ( origTileBBox.containsPoint( pixelCol, pixelRow ) ) {
											ByteProcessor origTileBp = origTiles[ origTileIdx ].render();
											int translatedX = pixelCol - origTileBBox.getStartPoint().getX();
											int translatedY = pixelRow - origTileBBox.getStartPoint().getY();
											if ( origTileBp.get( translatedX, translatedY ) != 0 ) {
												sumPixelValue += origTileBp.get( translatedX, translatedY );
												overlapPixelCount++;
											}
										}
									}
									
									if ( overlapPixelCount != 0 ) {
										targetBp.set(
												pixelCol - col,
												pixelRow - row,
												(sumPixelValue / overlapPixelCount) );
									}
								}
								
							}
															
							// Save the image to disk
							String outFile = outputDir + File.separatorChar + "tile_" + (row / tileSize) + "_" + (col / tileSize) + ".jpg";
							Utils.saveImage( targetImage, outFile, "jpg" );
							
							// Clear the output image
							raster.setDataElements( 0, 0, tileSize, tileSize, origData );
						}			
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
	}

	

	public static void main( final String[] args )
	{

		final Params params = parseParams( args );
		
		if ( params == null )
			return;
		
		
		/* open tilespec */
		final URL url;
		final TileSpec[] tileSpecs;
		try
		{
			final Gson gson = new Gson();
			url = new URL( params.url );
			tileSpecs = gson.fromJson( new InputStreamReader( url.openStream() ), TileSpec[].class );
		}
		catch ( final MalformedURLException e )
		{
			System.err.println( "URL malformed." );
			e.printStackTrace( System.err );
			return;
		}
		catch ( final JsonSyntaxException e )
		{
			System.err.println( "JSON syntax malformed." );
			e.printStackTrace( System.err );
			return;
		}
		catch ( final Exception e )
		{
			e.printStackTrace( System.err );
			return;
		}
		
		
		// Load each original tile and compute the entire image bbox
		System.out.println( "Loading tilespecs and computing entire bounding box" );
		SingleTileSpecRender[] origTilesRendered = new SingleTileSpecRender[ tileSpecs.length ];
		BoundingBox entireImageBBox = new BoundingBox();
		for ( int i = 0; i < origTilesRendered.length; i++ ) {
			origTilesRendered[ i ] = new SingleTileSpecRender( tileSpecs[ i ], params.numThreads );
			BoundingBox origTileBBox = origTilesRendered[ i ].getBoundingBox();
			// Update the entire image bounding box
			entireImageBBox.extendByBoundingBox(origTileBBox);
		}
		System.out.println( "Image bounding box is: " + entireImageBBox );
		
		// Pre-render all original tiles, so we can have a parallel version that saves
		// the new tiles
		System.out.println( "Rendering all tiles in the tilespec" );
		for ( int i = 0; i < origTilesRendered.length; i++ ) {
			origTilesRendered[ i ].render();
		}
		
		System.out.println( "Saving all tiles" );
		if ( params.numThreads == 1 ) {
			saveAsTiles( 
					origTilesRendered,
					entireImageBBox,
					params.tileSize, 
					params.targetDir );
		} else {
			saveAsTiles( 
					origTilesRendered,
					entireImageBBox,
					params.tileSize, 
					params.targetDir,
					params.numThreads );
		}
		
		System.out.println( "Done." );
		
	}

}

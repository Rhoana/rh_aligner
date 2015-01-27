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
import java.util.List;
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

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
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

//        @Parameter( names = "--blendType", description = "The type of image blending to use (BLEND_WEIGHTED_SIMPLE, BLEND_LINEAR)", required = false, converter = BlendTypeConverter.class, arity = 1 )
        @Parameter( names = "--blendType", description = "The type of image blending to use (BLEND_WEIGHTED_SIMPLE, BLEND_LINEAR)", required = false,  arity = 1 )
        public BlendType blendType = BlendType.BLEND_WEIGHTED_SIMPLE;

        @Parameter( names = "--outputNamePattern", description = "An output file name pattern where '%rowcol' will be replaced by '_tr[row]-tc[col]_' with the row and column numbers", required = false )
        public String outputNamePattern = "tile%rowcol";

        @Parameter( names = "--outputType", description = "The output image type", required = true )
        public String outputType = "jpg";

	}

	public enum BlendType
	{
		BLEND_WEIGHTED_SIMPLE,
		BLEND_LINEAR;

		// converter that will be used later
		public static BlendType fromString( String code ) {
			for ( BlendType type : BlendType.values() ) {
				if ( type.toString().equalsIgnoreCase( code ) ) {
					return type;
				}
			}

			return null;
		}
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
	
	private static void setOutputPixelBlendSimple(
			final SingleTileSpecRender[] origTiles,
			final ByteProcessor targetBp,
			final int globalPixelCol,
			final int globalPixelRow,
			final int tilePixelCol,
			final int tilePixelRow )
	{
		int sumPixelValue = 0;
		int overlapPixelCount = 0;
		
		// Iterate over all tiles that this pixel is part of, and the average of
		// this pixel's values will be used as the output value
		// Find the original tiles that are needed for the output tile
		for ( int origTileIdx = 0; origTileIdx < origTiles.length; origTileIdx++ ) {
			BoundingBox origTileBBox = origTiles[ origTileIdx ].getBoundingBox();
			if ( origTileBBox.containsPoint( globalPixelCol, globalPixelRow ) ) {
				ByteProcessor origTileBp = origTiles[ origTileIdx ].render();
				int translatedX = globalPixelCol - origTileBBox.getStartPoint().getX();
				int translatedY = globalPixelRow - origTileBBox.getStartPoint().getY();
				if ( origTileBp.get( translatedX, translatedY ) != 0 ) {
					sumPixelValue += origTileBp.get( translatedX, translatedY );
					overlapPixelCount++;
				}
			}
		}
		
		if ( overlapPixelCount != 0 ) {
			targetBp.set(
					tilePixelCol,
					tilePixelRow,
					(sumPixelValue / overlapPixelCount) );
		}

	}

	private static void setOutputPixelBlendLinear(
			final SingleTileSpecRender[] origTiles,
			final ByteProcessor targetBp,
			final int globalPixelCol,
			final int globalPixelRow,
			final int tilePixelCol,
			final int tilePixelRow )
	{
		List< Integer > pixelValuesList = new ArrayList< Integer >(); 
		List< Integer > pixelDistancesList = new ArrayList< Integer >();
		int sumDistance = 0;
		
		// Iterate over all tiles that this pixel is part of, and the average of
		// this pixel's values will be used as the output value
		// Find the original tiles that are needed for the output tile
		for ( int origTileIdx = 0; origTileIdx < origTiles.length; origTileIdx++ ) {
			BoundingBox origTileBBox = origTiles[ origTileIdx ].getBoundingBox();
			if ( origTileBBox.containsPoint( globalPixelCol, globalPixelRow ) ) {
				ByteProcessor origTileBp = origTiles[ origTileIdx ].render();
				int translatedX = globalPixelCol - origTileBBox.getStartPoint().getX();
				int translatedY = globalPixelRow - origTileBBox.getStartPoint().getY();
				pixelValuesList.add( origTileBp.get( translatedX, translatedY ) );
				// Add 1 to make sure that the distance is at least 1
				int minDist = origTileBBox.getMinimalL1DistanceToBBox( globalPixelCol, globalPixelRow ) + 1;
				pixelDistancesList.add( minDist );
				sumDistance = sumDistance + minDist;
			}
		}
		
		if ( pixelValuesList.size() == 1 ) {
			targetBp.set(
					tilePixelCol,
					tilePixelRow,
					pixelValuesList.get( 0 ) );
		} else if ( pixelValuesList.size() > 1 ) {
			double pixelValue = 0.0;
			for ( int i = 0; i < pixelValuesList.size(); i++ ) {
				// Normalize the pixel value according to the distance from each boundary
				pixelValue = pixelValue + ( pixelValuesList.get( i ) * ( (double)pixelDistancesList.get( i ) / sumDistance ) );
			}
			targetBp.set(
					tilePixelCol,
					tilePixelRow,
					( int )pixelValue );
		}

	}

	private static void setOutputPixel(
			final SingleTileSpecRender[] origTiles,
			final ByteProcessor targetBp,
			final int globalPixelCol,
			final int globalPixelRow,
			final int tilePixelCol,
			final int tilePixelRow,
			final BlendType blendType )
	{
		switch (blendType)
		{
		case BLEND_WEIGHTED_SIMPLE:
			setOutputPixelBlendSimple(
					origTiles, targetBp,
					globalPixelCol, globalPixelRow,
					tilePixelCol, tilePixelRow );
			break;
			
		case BLEND_LINEAR:
			setOutputPixelBlendLinear(
					origTiles, targetBp,
					globalPixelCol, globalPixelRow,
					tilePixelCol, tilePixelRow );
			break;
			
		default:
			System.err.println( "Unknown blending type selected" );	
		}
	}
	
	private static void saveAsTiles( 
			final String outFilePattern,
			final String outFileType,
			final SingleTileSpecRender[] origTiles,
			final BoundingBox entireImageBBox,
			final int tileSize,
			final String outputDir,
			final BlendType blendType )
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
						setOutputPixel( 
								origTiles, targetBp,
								pixelCol, pixelRow,
								pixelCol - col, pixelRow - row,
								blendType );

						/*
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
						*/
					}
					
				}
												
				// Save the image to disk
				String rowCol = outFilePattern.replaceAll( "%rowcol", "_tr" + (row / tileSize + 1) + "-tc" + (col / tileSize + 1) + "_" );
				String outFile = outputDir + File.separatorChar + rowCol + "." + outFileType;
				System.out.println( "Saving file: " + outFile );
				Utils.saveImage( targetImage, outFile, outFileType );
				
				// Clear the output image
				raster.setDataElements( 0, 0, tileSize, tileSize, origData );
			}			
		}
	}

	private static void saveAsTiles( 
			final String outFilePattern,
			final String outFileType,
			final SingleTileSpecRender[] origTiles,
			final BoundingBox entireImageBBox,
			final int tileSize,
			final String outputDir,
			final BlendType blendType,
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
									setOutputPixel( 
											origTiles, targetBp,
											pixelCol, pixelRow,
											pixelCol - col, pixelRow - row,
											blendType );

									/*
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
									*/
								}
								
							}
															
							// Save the image to disk
							String rowCol = outFilePattern.replaceAll( "%rowcol", "_tr" + (row / tileSize + 1) + "-tc" + (col / tileSize + 1) + "_" );
							String outFile = outputDir + File.separatorChar + rowCol + "." + outFileType;
							System.out.println( "Saving file: " + outFile );
							Utils.saveImage( targetImage, outFile, outFileType );
							
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
		
		if ( params.outputNamePattern.indexOf( "%rowcol" ) == -1 )
		{
			System.err.println( "outputNameFormat must have %rowcol in the pattern" );
			return;
		}

		
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
					params.outputNamePattern,
					params.outputType,
					origTilesRendered,
					entireImageBBox,
					params.tileSize, 
					params.targetDir,
					params.blendType );
		} else {
			saveAsTiles( 
					params.outputNamePattern,
					params.outputType,
					origTilesRendered,
					entireImageBBox,
					params.tileSize, 
					params.targetDir,
					params.blendType,
					params.numThreads );
		}
		
		System.out.println( "Done." );
		
	}

}

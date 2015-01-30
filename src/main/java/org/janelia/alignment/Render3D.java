package org.janelia.alignment;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

/**
 * Renders a given list of tile specs (each representing a layer) on the screen.
 * Scales the output image to fit some width.
 * It must receive all sections (all json files) in order to compute the entire image bounding box.
 */
public class Render3D {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

		// It must receive all sections (all json files) in order to compute the entire image bounding box.
		@Parameter(description = "Json files to render")
		private List<String> files = new ArrayList<String>();
		        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

        @Parameter( names = "--width", description = "The width of the output image (considering all sections)", required = false )
        public int width = 2500;

        @Parameter( names = "--fromLayer", description = "The layer to start the optimization from (default: first layer in the tile specs data)", required = false )
        private int fromLayer = -1;

        @Parameter( names = "--toLayer", description = "The last layer to include in the optimization (default: last layer in the tile specs data)", required = false )
        private int toLayer = -1;

        @Parameter( names = "--hide", description = "Hide the output and do not show on screen (default: false)", required = false )
        public boolean hide = false;

        @Parameter( names = "--targetDir", description = "The directory to save the output files to (default: no saving)", required = false )
        public String targetDir = null;

	}
	
	private Render3D() {}
		
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
		
		return params;
	}
	

	private static ImagePlus renderLayerImage( TileSpecsImage image, double scale, int layer, int entireWidth, int entireHeight )
	{
		System.out.println( "Showing layer: " + layer );
		
		ByteProcessor tp = image.render( layer, 0, (float) scale, entireWidth, entireHeight );
		ImagePlus curLayer = new ImagePlus( "Layer " + layer, tp );
		return curLayer;
	}

	private static void saveLayerImage( ImagePlus image, String outFile )
	{
		//IJ.saveAsTiff( image, outFile );
		//IJ.save( image, outFile );
		BufferedImage bufImage = image.getBufferedImage();
		Utils.saveImage( bufImage, outFile, outFile.substring( outFile.lastIndexOf( '.' ) + 1 ) );
		System.out.println( "Image " + outFile + " was saved." );
	}

	
	public static void main( final String[] args )
	{
		
		final Params params = parseParams( args );
		
		if ( params == null )
			return;

		if ( !params.hide )
			new ImageJ();

		List< String > actualTileSpecFiles;
		if ( params.files.size() == 1 )
			// It might be a non-json file that contains a list of
			actualTileSpecFiles = Utils.getListFromFile( params.files.get( 0 ) );
		else
			actualTileSpecFiles = params.files;

		final HashMap<Integer, String> layerIdToTileSpecFilename = new HashMap<Integer, String>();
		for ( String tsFileName : actualTileSpecFiles )
		{
			// read tile spec layer
			final TileSpec[] ts = TileSpecUtils.readTileSpecFile( tsFileName );
			if ( ( ts != null ) && ( ts.length > 0 ) )
				if ( ts[0].layer != -1 )
					layerIdToTileSpecFilename.put( ts[0].layer, tsFileName );
		}
		
		// Open all tiles
		final TileSpecsImage entireImage = TileSpecsImage.createImageFromFiles( actualTileSpecFiles );
		
		// Get the bounding box
		BoundingBox bbox = entireImage.getBoundingBox();
		
		// Get the entire image width and height (in case the
		// image starts in a place which is not (0,0), but (X, Y),
		// where X and Y are non-negative numbers)
		int entireImageWidth = bbox.getWidth();
		int entireImageHeight = bbox.getHeight();
		if ( bbox.getStartPoint().getX() > 0 )
			entireImageWidth += bbox.getStartPoint().getX();
		if ( bbox.getStartPoint().getY() > 0 )
			entireImageHeight += bbox.getStartPoint().getY();
		
		// Compute the initial scale (initialWidth pixels wide), round with a 2 digits position
		double scale;
		if ( params.width == -1 )
			scale = 1.0;
		else
		{
			scale = Math.round( ( (double)params.width / entireImageWidth ) * 100.0 ) / 100.0;
			scale = Math.min( scale, 1.0 );
		}
		
		System.out.println( "Scale is: " + scale );
		System.out.println( "Scaled width: " + (entireImageWidth * scale) + ", height: " + (entireImageHeight * scale) );
		
		// Render the first layer
		int firstLayer = params.fromLayer;
		if ( firstLayer == -1 )
			firstLayer = bbox.getStartPoint().getZ();

		int lastLayer = params.toLayer;
		if ( lastLayer == -1 )
			lastLayer = bbox.getEndPoint().getZ();

		// Render all needed layers
		if ( params.numThreads == 1 )
		{
			entireImage.setThreadsNum( 1 );
			
			// Single thread execution
			for ( int i = firstLayer; i <= lastLayer; i++ )
			{
				// If layer doesn't exist in the original files, just skip it
				if ( !layerIdToTileSpecFilename.containsKey( i ) )
					continue;

				final int curLayer = i;
				final double curScale = scale;

				final ImagePlus image = renderLayerImage( entireImage, curScale, curLayer, entireImageWidth, entireImageHeight );
				if ( !params.hide )
					image.show();
				if ( params.targetDir != null )
				{
					String jsonFilename = layerIdToTileSpecFilename.get( i );
					String baseFilename = jsonFilename.substring( jsonFilename.lastIndexOf( File.separatorChar ) + 1 );
					baseFilename = baseFilename.substring( 0, baseFilename.lastIndexOf( '.' ) );
					
					String outFile = String.format( "%04d_%s/%s.png", curLayer, params.targetDir, baseFilename );
					saveLayerImage( image, outFile );
				}
			}
		}
		else
		{
			entireImage.setThreadsNum( params.numThreads );
			
			// TODO: create a thread for saving the layer image (an IO thread)
			for ( int i = firstLayer; i <= lastLayer; i++ )
			{
				// If layer doesn't exist in the original files, just skip it
				if ( !layerIdToTileSpecFilename.containsKey( i ) )
					continue;

				final int curLayer = i;
				final double curScale = scale;

				final ImagePlus image = renderLayerImage( entireImage, curScale, curLayer, entireImageWidth, entireImageHeight );
				if ( !params.hide )
					image.show();
				if ( params.targetDir != null )
				{
					String jsonFilename = layerIdToTileSpecFilename.get( i );
					String baseFilename = jsonFilename.substring( jsonFilename.lastIndexOf( '/' ) + 1 );
					baseFilename = baseFilename.substring( 0, baseFilename.lastIndexOf( '.' ) );
					System.out.println( baseFilename );
					
					String outFile = String.format( "%04d_%s/%s.png", curLayer, params.targetDir, baseFilename );
					saveLayerImage( image, outFile );
				}
			}

		}
	}
}

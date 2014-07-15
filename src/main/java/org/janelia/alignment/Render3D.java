package org.janelia.alignment;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.process.ColorProcessor;

import java.util.ArrayList;
import java.util.List;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

/**
 * Renders a given list of tile specs (each representing a layer) on the screen.
 * Scales the output image to fit some width.
 * TODO: allow outputting the image to a movie.
 */
public class Render3D {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

		@Parameter(description = "Json files to render")
		private List<String> files = new ArrayList<String>();
		        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

        @Parameter( names = "--width", description = "The width of the output image (considering all sections)", required = false )
        public int width = 2500;

        @Parameter( names = "--layer", description = "The layer to render first (default: the first layer)", required = false )
        public int layer = -1;

        @Parameter( names = "--hide", description = "Hide the output and do not show on screen (default: false)", required = false )
        public boolean hide = false;

        @Parameter( names = "--targetDir", description = "The directory to save the output files to (default: no saving)", required = false )
        public String targetDir = null;

	}
	
	private Render3D() {}
	
	private static final int DEFAULT_LAYERS_NUM_TO_SHOW = 5;
	
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
	

	private static ImagePlus renderLayerImage( TileSpecsImage image, double scale, int layer )
	{
		System.out.println( "Showing layer: " + layer );
		
		ColorProcessor cp = image.render( layer, 0, (float) scale );
		ImagePlus curLayer = new ImagePlus( "Layer " + layer, cp );
		return curLayer;
	}

	private static void saveLayerImage( ImagePlus image, String outFile )
	{
		IJ.save( image, outFile );
		System.out.println( "Image " + outFile + " was saved." );
	}

	
	public static void main( final String[] args )
	{
		
		final Params params = parseParams( args );
		
		if ( params == null )
			return;

		if ( !params.hide )
			new ImageJ();

		// Open all tiles
		TileSpecsImage entireImage = TileSpecsImage.createImageFromFiles( params.files );
		
		// Get the bounding box
		BoundingBox bbox = entireImage.getBoundingBox();
		
		// Compute the initial scale (initialWidth pixels wide), round with a 2 digits position
		double scale = Math.round( ( (double)params.width / bbox.getWidth() ) * 100.0 ) / 100.0;
		scale = Math.min( scale, 1.0 );
		
		System.out.println( "Scale is: " + scale );
		System.out.println( "Scaled width: " + (bbox.getWidth() * scale) + ", height: " + (bbox.getHeight() * scale) );

		// Render the first layer
		int firstLayer = params.layer;
		if ( firstLayer == -1 )
			firstLayer = bbox.getStartPoint().getZ();

		// if no layer is given as input, show all layers
		if ( params.layer == -1 )
		{
			final int lastLayer = bbox.getEndPoint().getZ();
			for ( int i = firstLayer; i < lastLayer + 1; i++ )
			{
				ImagePlus image = renderLayerImage( entireImage, scale, i );
				if ( !params.hide )
					image.show();
				if ( params.targetDir != null )
				{
					String outFile = String.format( "%s/Section_%03d.png", params.targetDir, i );
					saveLayerImage( image, outFile );
				}
			}
		}
		else // Show the wanted layer
		{
			ImagePlus image = renderLayerImage( entireImage, scale, firstLayer );
			if ( !params.hide )
				image.show();
			if ( params.targetDir != null )
			{
				String outFile = String.format( "%s/Section_%03d.png", params.targetDir, firstLayer  );
				saveLayerImage( image, outFile );
			}
		}
		
		/* save the modified image */
		/*
		if (params.out != null)
		{
			System.out.print("Saving output image to disk...   ");
			Utils.saveImage( targetImage, params.out, params.out.substring( params.out.lastIndexOf( '.' ) + 1 ), params.quality );
			System.out.println("Done.");
			//new ImagePlus( params.out ).show();
		}
		*/
	}
}

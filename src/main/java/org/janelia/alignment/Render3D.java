package org.janelia.alignment;

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

        @Parameter( names = "--initial_width", description = "The initial width of the image", required = false )
        public int initialWidth = 2500;

        @Parameter( names = "--layer", description = "The layer to render first (default: the first layer)", required = false )
        public int layer = -1;
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
	
	private static void showLayerImage( TileSpecsImage image, double scale, int layer )
	{
		System.out.println( "Showing layer: " + layer );
		
		ColorProcessor cp = image.render( layer, 0, (float) scale );
		ImagePlus curLayer = new ImagePlus( "Layer " + layer, cp );
		curLayer.show();
	}
	
	public static void main( final String[] args )
	{
		new ImageJ();
		
		final Params params = parseParams( args );
		
		if ( params == null )
			return;
		
		// Open all tiles
		TileSpecsImage entireImage = TileSpecsImage.createImageFromFiles( params.files );
		
		// Get the bounding box
		BoundingBox bbox = entireImage.getBoundingBox();
		
		// Compute the initial scale (initialWidth pixels wide), round with a 2 digits position
		double scale = Math.round( ( (double)params.initialWidth / bbox.getWidth() ) * 100.0 ) / 100.0;
		scale = Math.min( scale, 1.0 );
		
		System.out.println( "Scale is: " + scale );
		System.out.println( "Scaled width: " + (bbox.getWidth() * scale) + ", height: " + (bbox.getHeight() * scale) );

		// Render the first layer
		int firstLayer = params.layer;
		if ( firstLayer == -1 )
			firstLayer = bbox.getStartPoint().getZ();

		// if no layer is given as input, show the first DEFAULT_LAYERS_NUM_TO_SHOW layers
		if ( params.layer == -1 )
		{
			final int lastLayer = Math.min( bbox.getStartPoint().getZ() + bbox.getDepth(),
					firstLayer + DEFAULT_LAYERS_NUM_TO_SHOW );
			for ( int i = firstLayer; i < lastLayer; i++ )
			{
				showLayerImage( entireImage, scale, i );
			}
		}
		else // Show the wanted layer
			showLayerImage( entireImage, scale, firstLayer );
		
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

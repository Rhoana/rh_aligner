package org.janelia.alignment;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

/**
 * Saves the mesh after applying the transformations (useful for large images)
 *  
 * @author Adi Suissa-Peleg
 *
 */
public class SaveMesh {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

		// It must receive all sections (all json files) in order to compute the entire image bounding box.
		@Parameter( names = "--inputfile", description = "TileSpec json file which we wish to save mesh for each of its tiles")
		private String inputfile;
		        
        @Parameter( names = "--targetDir", description = "The directory to save the output files to", required = true )
        public String targetDir;

        @Parameter( names = "--threads", description = "Number of threads to use", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

	}
	
	private SaveMesh() {}
		
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
        	throw new RuntimeException( e );
        }
		
		return params;
	}
	

	public static void main( final String[] args )
	{
		
		final Params params = parseParams( args );
		
		if ( params == null )
			return;
		
		// Open all tiles
		final TileSpecsImage entireImage = TileSpecsImage.createImageFromFile( params.inputfile );
		
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
		
		double scale = 1.0;
		
		System.out.println( "Scale is: " + scale );
		System.out.println( "Scaled width: " + (entireImageWidth * scale) + ", height: " + (entireImageHeight * scale) );
		
		entireImage.setThreadsNum( params.numThreads );

		// Save the mesh of all tiles
		entireImage.saveMeshes( params.targetDir, bbox.getStartPoint().getZ(), 0 );
		
	}

}

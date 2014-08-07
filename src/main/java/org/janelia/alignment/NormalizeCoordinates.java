package org.janelia.alignment;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mpicbg.trakem2.transform.TranslationModel2D;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * Normalizes (shifts/translates) the coordinates of a given 3D tile spec image to
 * the (0, 0) coordinates.
 * Outputs the tile spec to a file with the same name in a given output directory
 */
public class NormalizeCoordinates {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

		@Parameter(description = "Json files of a single 3D image to normalize coordinates for")
		private List<String> files = new ArrayList<String>();
        
		@Parameter( names = "--targetDir", description = "The directory where the output json files will be saved (SectionNNN.json)", required = true )
		public String targetDir;
		        
	}

	private NormalizeCoordinates() { }
	
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
	
	
	public static void main( final String[] args )
	{		
		final Params params = parseParams( args );
		
		if ( params == null )
			return;
		
		final TileSpecsImage tsImage = TileSpecsImage.createImageFromFiles( params.files );

		// Get the bounding box
		final BoundingBox bbox = tsImage.getBoundingBox();
		System.out.println( "Bounding box is: " + bbox );
		
		// If the image does not start in (0, 0)
		final int curX = bbox.getStartPoint().getX();
		final int curY = bbox.getStartPoint().getY();
		if ( ( curX != 0 ) || ( curY != 0 ) )
		{
			System.out.println( "Translating by: (" + -curX + "," + -curY + ")" );
			// Create a transformation to be added to all tile specs
			TranslationModel2D trans = new TranslationModel2D();
			trans.init( -curX + ".0 " + -curY + ".0" );

			for ( String fileName : params.files )
			{
				// Read the json file
				TileSpec[] tileSpecs = TileSpecUtils.readTileSpecFile( fileName );
				
				// Add the transformation to each tile spec
				for ( final TileSpec ts : tileSpecs )
				{
				    Transform addedTransform = new Transform();
				    addedTransform.className = trans.getClass().getCanonicalName().toString();
				    addedTransform.dataString = trans.toDataString();

					final ArrayList< Transform > outTransforms = new ArrayList< Transform >(Arrays.asList(ts.transforms));
					outTransforms.add( addedTransform );
					ts.transforms = outTransforms.toArray( ts.transforms );
					
					// Update the bounding box of the tile
					if ( ts.bbox != null )
					{
						ts.bbox[0] = ts.bbox[0] - curX;
						ts.bbox[1] = ts.bbox[1] - curX;
						ts.bbox[2] = ts.bbox[2] - curY;
						ts.bbox[3] = ts.bbox[3] - curY;
					}
				}
				
				// Save the output file
				final String outFileName = params.targetDir + fileName.substring( fileName.lastIndexOf(File.separatorChar) );
				System.out.println( "Normalizing " + fileName + " to " + outFileName );
				try {
					Writer writer = new FileWriter( outFileName );
			        Gson gson = new GsonBuilder().setPrettyPrinting().create();
			        gson.toJson( tileSpecs, writer );
			        writer.close();
			    }
				catch ( final IOException e )
				{
					System.err.println( "Error writing JSON file: " + outFileName );
					e.printStackTrace( System.err );
				}

			}

		}
		
	}

}

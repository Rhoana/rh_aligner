package org.janelia.alignment;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.List;

import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.trakem2.transform.AffineModel2D;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

public class DebugFilterRansac {
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--inputfile", description = "The url of the filter ransac output json file", required = true )
        private String inputfile;

        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        private float layerScale = 0.1f;
        
        @Parameter( names = "--targetDir", description = "The output directory", required = true )
        private String targetDir;

        @Parameter( names = "--showResults", description = "Whether to show the images that are created", required = false )
        private boolean showResults = false;

	}

	private DebugFilterRansac() {}

	
	private static void saveImage(
			final String tileSpecUrl,
			final float scale,
			final int mipmapLevel,
			final CoordinateTransform model,
			final String targetPath,
			final boolean showResults )
	{
		// Load tile specs from the tileSpecUrl
		TileSpec[] tilespecs = TileSpecUtils.readTileSpecFile( tileSpecUrl );
		
		// Add the model (if there is one)
		if ( model != null )
		{
			System.out.println( "Adding test" );
			for ( TileSpec ts : tilespecs )
			{
				CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
				ctl.add( model );
				
				List< CoordinateTransform > lst = ctl.getList( null );
				ts.transforms = new Transform[ lst.size() ];
				for ( int j = 0; j < lst.size(); j++ )
				{
					CoordinateTransform tran = lst.get( j );
					ts.transforms[ j ] = Transform.createTransform( tran );
				}
			}
		}
		
		// Render the image
		final TileSpecsImage singleTileImage = new TileSpecsImage( tilespecs );
		final BoundingBox bbox = singleTileImage.getBoundingBox();
		final int layerIndex = bbox.getStartPoint().getZ();
		ByteProcessor bp = singleTileImage.render( layerIndex, mipmapLevel, scale );
		ColorProcessor cp = bp.convertToColorProcessor();

		System.out.println( "The output image size: " + cp.getWidth() + ", " + cp.getHeight() );

		final BufferedImage image = new BufferedImage( cp.getWidth(), cp.getHeight(), BufferedImage.TYPE_INT_RGB );
		final WritableRaster raster = image.getRaster();
		raster.setDataElements( 0, 0, cp.getWidth(), cp.getHeight(), cp.getPixels() );
		
		// Save the rendered image
		if ( showResults )
			new ImagePlus( "debug", cp ).show();

		final BufferedImage targetImage = new BufferedImage( cp.getWidth(), cp.getHeight(), BufferedImage.TYPE_INT_RGB );
		final Graphics2D targetGraphics = targetImage.createGraphics();
		targetGraphics.drawImage( image, 0, 0, null );
		
		String fileName = targetPath + ".png";
		Utils.saveImage( targetImage, fileName, fileName.substring( fileName.lastIndexOf( '.' ) + 1 ) );

	}
	
	public static void main( final String[] args )
	{

		final Params params = new Params();
		try
        {
			final JCommander jc = new JCommander( params, args );
        	if ( params.help )
            {
        		jc.usage();
                return;
            }
        }
        catch ( final Exception e )
        {
        	e.printStackTrace();
            final JCommander jc = new JCommander( params );
        	jc.setProgramName( "java [-options] -cp render.jar org.janelia.alignment.RenderTile" );
        	jc.usage();
        	return;
        }

		// The mipmap level to work on
		// TODO: Should be a parameter from the user,
		//       and decide whether or not to create the mipmaps if they are missing
		int mipmapLevel = 0;

		/* open the models */
		CoordinateTransform model = null;
		final ModelSpec[] modelSpecs;
		try
		{
			final Gson gson = new Gson();
			URL url = new URL( params.inputfile );
			modelSpecs = gson.fromJson( new InputStreamReader( url.openStream() ), ModelSpec[].class );			
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

		final String inFileName = params.inputfile.substring( params.inputfile.lastIndexOf( '/' ) + 1 );
		for ( int i = 0; i < modelSpecs.length; i++)
		{
			final ModelSpec ms = modelSpecs[i];
			model = ms.createModel();

			if ( model == null )
			{
				System.out.println( "Given model is null, skipping model");
			}
			else
			{
				final String tilespec1 = ms.url1;
				final String tilespec2 = ms.url2;

				// Save the first image
				final String outputFile1 = params.targetDir + File.separatorChar + inFileName + "_entry" + i + "_0";
				saveImage( tilespec1, params.layerScale, mipmapLevel, null, outputFile1, params.showResults );
				
				// Save the second image
				final String outputFile2 = params.targetDir + File.separatorChar + inFileName + "_entry" + i + "_1";
				saveImage( tilespec2, params.layerScale, mipmapLevel, model, outputFile2, params.showResults );
			}

		}

		System.out.println( "Done." );
	}

}

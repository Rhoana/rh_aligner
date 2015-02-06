package org.janelia.alignment;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;

import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import mpicbg.ij.SIFT;
import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.FloatArray2DSIFT;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

public class ComputeLayerSiftFeatures {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--url", description = "URL to JSON tile spec", required = true )
        private String url;

        @Parameter( names = "--meshesDir", description = "The directory where the cached mesh per tile is located", required = false )
        private String meshesDir = null;

        @Parameter( names = "--initialSigma", description = "Initial Gaussian blur sigma", required = false )
        private float initialSigma = 1.6f;
        
        @Parameter( names = "--steps", description = "Steps per scale octave", required = false )
        private int steps = 3;
        
        @Parameter( names = "--minOctaveSize", description = "Min image size", required = false )
        private int minOctaveSize = 64;
        
        @Parameter( names = "--maxOctaveSize", description = "Max image size", required = false )
        private int maxOctaveSize = 1024;
        
        @Parameter( names = "--fdSize", description = "Feature descriptor size", required = false )
        private int fdSize = 8;
        
        @Parameter( names = "--fdBins", description = "Feature descriptor bins", required = false )
        private int fdBins = 8;
        
        @Parameter( names = "--targetPath", description = "Path to the target image if any", required = true )
        public String targetPath = null;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
//        @Parameter( names = "--res", description = " Mesh resolution, specified by the desired size of a triangle in pixels", required = false )
//        public int res = 64;

        @Parameter( names = "--scale", description = "Layer scale (if given, avoids automatic layer scale)", required = false )
        public float scale = -1.0f;

	}
	
	private ComputeLayerSiftFeatures() {}
/*	
	private static float[] translateBoundingBox( float[] originalBoundingBox, CoordinateTransformList< CoordinateTransform > ctlMipmap )
	{
		float[][] points = {
				{ originalBoundingBox[0], originalBoundingBox[2] },
				{ originalBoundingBox[0], originalBoundingBox[3] },
				{ originalBoundingBox[0], originalBoundingBox[2] },
		};
	
	}
*/
	
    public static List< Feature > computeTileSiftFeatures( String imageUrl, FloatArray2DSIFT.Param siftParam )
    {
            /* calculate sift features for the image or sub-region */
            System.out.println( "Calculating SIFT features for image '" + imageUrl + "'." );
            final ImagePlus imp = Utils.openImagePlus( imageUrl.replaceFirst("file://", "").replaceFirst("file:/", "") );
            if ( imp == null )
            {
                    throw new RuntimeException( "Failed to load image '" + imageUrl );
            }

            FloatArray2DSIFT sift = new FloatArray2DSIFT(siftParam);
            SIFT ijSIFT = new SIFT(sift);


            final List< Feature > fs = new ArrayList< Feature >();
            ijSIFT.extractFeatures( imp.getProcessor(), fs );

            return fs;
    }

    public static List< Feature > computeImageSiftFeatures( ImageProcessor ip, FloatArray2DSIFT.Param siftParam )
    {
            FloatArray2DSIFT sift = new FloatArray2DSIFT(siftParam);
            SIFT ijSIFT = new SIFT(sift);


            final List< Feature > fs = new ArrayList< Feature >();
            ijSIFT.extractFeatures( ip, fs );

            return fs;
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
						
		List< FeatureSpec > feature_data = new ArrayList< FeatureSpec >();
		
		// The mipmap level to work on
		// TODO: Should be a parameter from the user,
		//       and decide whether or not to create the mipmaps if they are missing
		int mipmapLevel = 0;

		/* calculate sift features for the image or sub-region */
		FloatArray2DSIFT.Param siftParam = new FloatArray2DSIFT.Param();
		siftParam.initialSigma = params.initialSigma;
		siftParam.steps = params.steps;
		siftParam.minOctaveSize = params.minOctaveSize;
		siftParam.maxOctaveSize = params.maxOctaveSize;
		siftParam.fdSize = params.fdSize;
		siftParam.fdBins = params.fdBins;
	
		// Create the layer image
		final TileSpecsImage singleTileImage = TileSpecsImage.createImageFromFile( params.url );
		singleTileImage.setThreadsNum( params.numThreads );
		final BoundingBox bbox = singleTileImage.getBoundingBox();
		final int layerIndex = bbox.getStartPoint().getZ();
		
		final double scale;
		if ( params.scale != -1.0f )
			scale = params.scale;
		else
			scale = Math.min( 1.0, Math.min( ( double )params.maxOctaveSize / ( double )bbox.getWidth(), ( double )params.maxOctaveSize / ( double )bbox.getHeight() ) );
		

		// Render the image
		System.out.println( "Sift Features computation: layer scale: " + scale );
		ByteProcessor tp;
		if ( params.meshesDir == null )
			tp = singleTileImage.render( layerIndex, mipmapLevel, ( float )scale );
		else
			tp = singleTileImage.renderFromMeshes( params.meshesDir, layerIndex, mipmapLevel, ( float )scale );
		System.out.println( "Image rendering of layer " + layerIndex + " is done, computing sift features." );
		final List< Feature > fs = computeImageSiftFeatures( tp, siftParam );
		System.out.println( "Found " + fs.size() + " features in the layer" );

		//final List< Feature > fs = computeTileSiftFeatures( imageUrl, siftParam );

		/* Apply the transformations on the location of every feature */
//			final CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
//			for (Feature feature : fs)
//			{
//				ctl.applyInPlace(feature.location);				
//			}

		feature_data.add(new FeatureSpec( String.valueOf( mipmapLevel ), params.url, scale, fs));

		if ( feature_data.size() > 0 )
		{
			try {
				Writer writer = new FileWriter(params.targetPath);
				//Gson gson = new GsonBuilder().create();
				Gson gson = new GsonBuilder().setPrettyPrinting().create();
				gson.toJson(feature_data, writer);
				writer.close();
			}
			catch ( final IOException e )
			{
				System.err.println( "Error writing JSON file: " + params.targetPath );
				e.printStackTrace( System.err );
			}
		}
	}
}

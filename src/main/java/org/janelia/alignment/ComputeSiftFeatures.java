/**
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package org.janelia.alignment;

import ij.ImageJ;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.io.FileWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.FloatArray2DSIFT;
import mpicbg.ij.SIFT;
import mpicbg.ij.clahe.Flat;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.CoordinateTransformMesh;
import mpicbg.models.TransformMesh;
import mpicbg.trakem2.transform.TransformMeshMapping;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks.ImageProcessorWithMasks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

/**
 * 
 * @author Seymour Knowles-Barley
 */
public class ComputeSiftFeatures
{
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--url", description = "URL to JSON tile spec", required = true )
        private String url;

        @Parameter( names = "--index", description = "Tile index", required = false )
        private int index = 0;

        
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
        
        @Parameter( names = "--targetPath", description = "Path to the target json file", required = true )
        public String targetPath;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
        @Parameter( names = "--res", description = " Mesh resolution, specified by the desired size of a triangle in pixels", required = false )
        public int res = 64;

        @Parameter( names = "--useClaheFilter", description = "useClaheFilter", required = false )
        public boolean useClaheFilter = false;

        @Parameter( names = "--claheBlockSize", description = "The block size in pixels for the CLAHE filter", required = false )
        public int claheBlockSize = 127;

        @Parameter( names = "--claheHistBins", description = "The number of histogram bins for the CLAHE filter", required = false )
        public int claheHistBins = 256;

        @Parameter( names = "--claheMaxSlope", description = "The maximum slope for the CLAHE filter (lower number -> less filtering)", required = false )
        public float claheMaxSlope = 3.0f;

        @Parameter( names = "--claheFast", description = "Use a fast CLAHE filter (faster)", required = false )
        public boolean claheFast = false;

        @Parameter( names = "--distanceFromBoundariesPercent", description = "Only save features that are located in a given L1 distance from one of the image boundaries (default: any distance)", required = false )
        public float distanceFromBoundariesPercent = 0f;

        /*
        @Parameter( names = "--avoidTileScale", description = "Avoid automatic scale of all tiles according to the bounding box width and height", required = false )
        private boolean avoidTileScale = false;
        */
        
        @Parameter( names = "--minFeaturesNum", description = "Minimum number of features (after boundary filtering) that need to be in a tile in order to save these features", required = false )
        public int minFeaturesNum = 0;

	}
	
	private ComputeSiftFeatures() {}
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
	
	
	private static ByteProcessor createMaskedByteImage( ImageProcessor imgP )
	{
		
		/*
		final ByteProcessor mask;
		final Patch.PatchImage pai = patch.createTransformedImage();
		
		if ( pai.mask == null )
			mask = pai.outside;
		else
			mask = pai.mask;

		pai.target.setMinAndMax( patch.getMin(), patch.getMax() );
		*/
		
		//final ByteProcessor target = ( ByteProcessor )pai.target.convertToByte( true );
		final ByteProcessor target = ( ByteProcessor )imgP.convertToByte( true );

		/* Other than any other ImageProcessor, ByteProcessors ignore scaling, so ... */
		if ( ByteProcessor.class.isInstance( target ) )
		{
			final float s = 255.0f / ( float )( imgP.getMax() - imgP.getMin() );
			final int m = ( int )imgP.getMin();
			final byte[] targetBytes = ( byte[] )target.getPixels();
			for ( int i = 0; i < targetBytes.length; ++i )
			{
				targetBytes[ i ] = ( byte )( Math.max( 0, Math.min( 255, ( ( targetBytes[ i ] & 0xff ) - m ) * s ) ) );
			}
			target.setMinAndMax( 0, 255 );
		}

		/*
		if ( mask != null )
		{
			final byte[] targetBytes = ( byte[] )target.getPixels();
			final byte[] maskBytes = (byte[])mask.getPixels();

			if ( pai.outside != null )
			{
				final byte[] outsideBytes = (byte[])pai.outside.getPixels();
				for ( int i = 0; i < outsideBytes.length; ++i )
				{
					if ( ( outsideBytes[ i ]&0xff ) != 255 ) maskBytes[ i ] = 0;
					final float a = ( float )( maskBytes[ i ] & 0xff ) / 255f;
					final int t = ( targetBytes[ i ] & 0xff );
					targetBytes[ i ] = ( byte )( t * a + 127 * ( 1 - a ) );
				}
			}
			else
			{
				for ( int i = 0; i < targetBytes.length; ++i )
				{
					final float a = ( float )( maskBytes[ i ] & 0xff ) / 255f;
					final int t = ( targetBytes[ i ] & 0xff );
					targetBytes[ i ] = ( byte )( t * a + 127 * ( 1 - a ) );
				}
			}
		}
		*/
		
		return target;
	}
	
	public static List< Feature > computeImageSiftFeatures( ImageProcessor ip, FloatArray2DSIFT.Param siftParam )
	{
		FloatArray2DSIFT sift = new FloatArray2DSIFT(siftParam);
		SIFT ijSIFT = new SIFT(sift);


		final List< Feature > fs = new ArrayList< Feature >();
		ijSIFT.extractFeatures( createMaskedByteImage( ip ), fs );

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

        ij.Prefs.setThreads( params.numThreads );

		// The mipmap level to work on
		// TODO: Should be a parameter from the user,
		//       and decide whether or not to create the mipmaps if they are missing
		int mipmapLevel = 0;

		int start_index = params.index;
		int end_index = params.index + 1;

		
        final double scale;
        /*
        if ( params.avoidTileScale )
                scale = 1.0f;
        else
                scale = Math.min( 1.0, Math.min( ( double )params.maxOctaveSize / ( double )maxWidth, ( double )params.maxOctaveSize / ( double )maxHeight ) );
		*/
        scale = 1.0f;

		// Get the maximal width and height of all tiles (for resizing them later if needed)
		double maxWidth = 0.0;
		double maxHeight = 0.0;
        if ( scale != 1.0f ) {
			for (int idx = start_index; idx < end_index; idx = idx + 1) {
				TileSpec ts = tileSpecs[idx];
	
				String imageUrl = ts.getMipmapLevels().get( String.valueOf( mipmapLevel ) ).imageUrl;
				final ImagePlus imp = Utils.openImagePlus( imageUrl.replaceFirst("file://", "").replaceFirst("file:/", "") );
				if ( imp == null )
					System.err.println( "Failed to load image '" + imageUrl + "'." );
				else {
					maxWidth = Math.max( maxWidth, imp.getWidth() );
					maxHeight = Math.max( maxHeight, imp.getHeight() );
				}
			}
        }
        
		//new ImageJ();

		for (int idx = start_index; idx < end_index; idx = idx + 1) {
			TileSpec ts = tileSpecs[idx];
		
			/* load image TODO use Bioformats for strange formats */
			String imageUrl = ts.getMipmapLevels().get( String.valueOf( mipmapLevel ) ).imageUrl;
			final ImagePlus imp = Utils.openImagePlus( imageUrl.replaceFirst("file://", "").replaceFirst("file:/", "") );
			if ( imp == null )
				System.err.println( "Failed to load image '" + imageUrl + "'." );
			else
			{
				/* calculate sift features for the image or sub-region */
				System.out.println( "Calculating SIFT features for image '" + imageUrl + "'." );
				FloatArray2DSIFT.Param siftParam = new FloatArray2DSIFT.Param();
				siftParam.initialSigma = params.initialSigma;
				siftParam.steps = params.steps;
				siftParam.minOctaveSize = params.minOctaveSize;
				siftParam.maxOctaveSize = params.maxOctaveSize;
				siftParam.fdSize = params.fdSize;
				siftParam.fdBins = params.fdBins;
				//FloatArray2DSIFT sift = new FloatArray2DSIFT(siftParam);
				//SIFT ijSIFT = new SIFT(sift);
		
//			/* Apply transformations on the image, and only then get the sift features
//			 * (transformations may have an impact on the sift features)
//			 */
//			final ImageProcessor ipProc = imp.getProcessor();
//			final CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
//			/* attach mipmap transformation */
//			final CoordinateTransformList< CoordinateTransform > ctlMipmap = new CoordinateTransformList< CoordinateTransform >();
//			ctlMipmap.add( Utils.createScaleLevelTransform( mipmapLevel ) );
//			ctlMipmap.add( ctl );
//			/* find bounding box after transformations */
//			float[] originalBoundingBox = { 0, ipProc.getWidth(), 0, ipProc.getHeight() }; // left, right, top, bottom
//			float[] newBoundingBox = translateBoundingBox( originalBoundingBox, ctlMipmap );
//			/* create mesh */
//			final CoordinateTransformMesh mesh = new CoordinateTransformMesh( ctlMipmap,  ( int )( ipProc.getWidth() / params.res + 0.5 ), ipProc.getWidth(), ipProc.getHeight() );
//			final ImageProcessorWithMasks source = new ImageProcessorWithMasks( ipProc, null, null ); // no mask
//			final ImageProcessor tp = ipProc.createProcessor( targetImage.getWidth(), targetImage.getHeight() );
//			final ImageProcessorWithMasks target = new ImageProcessorWithMasks( tp, null, null ); // no mask
//			final TransformMeshMappingWithMasks< TransformMesh > mapping = new TransformMeshMappingWithMasks< TransformMesh >( mesh );
//			mapping.mapInterpolated( source, target );
//
//			
//			/* create mesh */
		
		
                System.out.println( "Sift Features computation: tile scale: " + scale );
                ImageProcessor scaledImp;
                if ( scale == 1.0f )
                	scaledImp = imp.getProcessor();
                else
                	scaledImp = imp.getProcessor().resize( (int)(maxWidth * scale), (int)(maxHeight * scale) );
                
				if ( params.useClaheFilter )
				{
					System.out.println( "Applying CLAHE filter" );
					//new ImagePlus("before", scaledImp).show();
					
					if ( params.claheFast )
						Flat.getFastInstance().run( new ImagePlus("", scaledImp), params.claheBlockSize, params.claheHistBins, params.claheMaxSlope, null, false);
					else
						Flat.getInstance().run( new ImagePlus("", scaledImp), params.claheBlockSize, params.claheHistBins, params.claheMaxSlope, null, false);
					System.out.println( "Applying CLAHE filter - Done" );
					//new ImagePlus("after", scaledImp).show();
				}

                final List< Feature > fs = ComputeSiftFeatures.computeImageSiftFeatures( scaledImp, siftParam );
                System.out.println( "Found " + fs.size() + " features in the tile" );
				//ijSIFT.extractFeatures( imp.getProcessor(), fs );
		
				/* Apply the transformations on the location of every feature */
/*				final CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
				for (Feature feature : fs)
				{
					ctl.applyInPlace(feature.location);				
				}*/
		
                // Filter out features that are not close to the boundary
                if ( params.distanceFromBoundariesPercent != 0f )
                {
                	float[] leftRightX = { 0f, 0f }; 
                	float[] topBottomY = { 0f, 0f }; 
                	if ( scale == 1.0f )
                	{
                		// Set X
                		leftRightX[0] = params.distanceFromBoundariesPercent * imp.getWidth();
                		leftRightX[1] = imp.getWidth() - params.distanceFromBoundariesPercent * imp.getWidth();

                		// Set Y
                		topBottomY[0] = params.distanceFromBoundariesPercent * imp.getHeight();
                		topBottomY[1] = imp.getWidth() - params.distanceFromBoundariesPercent * imp.getHeight();
                	}
                	else
                	{
	                	int scaledImageWidth = (int)(maxWidth * scale);
	                	int scaledImageHeight = (int)(maxHeight * scale);
                		// Set X
                		leftRightX[0] = params.distanceFromBoundariesPercent * scaledImageWidth;
                		leftRightX[1] = scaledImageWidth - params.distanceFromBoundariesPercent * scaledImageWidth;

                		// Set Y
                		topBottomY[0] = params.distanceFromBoundariesPercent * scaledImageHeight;
                		topBottomY[1] = scaledImageHeight - params.distanceFromBoundariesPercent * scaledImageHeight;
                	}
                	
                	Iterator<Feature> it = fs.iterator();
                	while ( it.hasNext() )
                	{
                		Feature f = it.next();
                		float[] location = f.location;
                		if ( ( location[0] > leftRightX[0] ) && ( location[0] < leftRightX[1] ) &&
                			 ( location[1] > topBottomY[0] ) && ( location[1] < topBottomY[1] ) )
                			it.remove();
                	}
                	
                    System.out.println( "Found " + fs.size() + " features in the tile after filtering non-boundary matches (" + params.distanceFromBoundariesPercent + ")" );
                }
                
        		if ( fs.size() >= params.minFeaturesNum )
        		{
        			feature_data.add(new FeatureSpec( String.valueOf( mipmapLevel ), imageUrl, scale, fs));
        		}
			}
		}

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
		System.out.println( "Done" );
	}
}

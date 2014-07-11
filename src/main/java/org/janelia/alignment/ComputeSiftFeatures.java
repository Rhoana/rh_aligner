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

import ij.ImagePlus;
import ij.process.ImageProcessor;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.io.FileWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.FloatArray2DSIFT;
import mpicbg.ij.SIFT;
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

        @Parameter( names = "--all", description = "Compute for all tiles", required = false )
        private boolean all_tiles = true;
        
        
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
        
        @Parameter( names = "--res", description = " Mesh resolution, specified by the desired size of a triangle in pixels", required = false )
        public int res = 64;

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

		int start_index = params.all_tiles ? 0 : params.index;
		int end_index = params.all_tiles ? tileSpecs.length : params.index + 1;

		/* calculate sift features for the image or sub-region */
		FloatArray2DSIFT.Param siftParam = new FloatArray2DSIFT.Param();
		siftParam.initialSigma = params.initialSigma;
		siftParam.steps = params.steps;
		siftParam.minOctaveSize = params.minOctaveSize;
		siftParam.maxOctaveSize = params.maxOctaveSize;
		siftParam.fdSize = params.fdSize;
		siftParam.fdBins = params.fdBins;

		for (int idx = start_index; idx < end_index; idx = idx + 1) {
			TileSpec ts = tileSpecs[idx];
		
			/* load image TODO use Bioformats for strange formats */
			String imageUrl = ts.getMipmapLevels().get( String.valueOf( mipmapLevel ) ).imageUrl;

		
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
		

			final List< Feature > fs = computeTileSiftFeatures( imageUrl, siftParam );
	
			/* Apply the transformations on the location of every feature */
			final CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
			for (Feature feature : fs)
			{
				ctl.applyInPlace(feature.location);				
			}
	
			final double scale = 1.0;
			
			feature_data.add(new FeatureSpec( String.valueOf( mipmapLevel ), imageUrl, scale, fs));
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
	}
}

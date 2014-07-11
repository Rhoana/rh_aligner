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

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.io.FileWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import mpicbg.models.AbstractModel;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.Transforms;
import mpicbg.trakem2.transform.AffineModel2D;
import mpicbg.trakem2.transform.CoordinateTransform;
import mpicbg.trakem2.transform.HomographyModel2D;
import mpicbg.trakem2.transform.RigidModel2D;
import mpicbg.trakem2.transform.SimilarityModel2D;
import mpicbg.trakem2.transform.TranslationModel2D;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

/** 
 * @author Seymour Knowles-Barley
 */
public class FilterRansac
{
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--inputfile", description = "Correspondence list file", required = true )
        private String inputfile;
                        
        @Parameter( names = "--comparedUrl", description = "The tilespec url to compare all correspondences with", required = true )
        private String compareUrl;

        @Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;
        
        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        private float layerScale = 0.1f;
        
        @Parameter( names = "--modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int modelIndex = 3;
        
        //@Parameter( names = "--regularizerIndex", description = "Regularizer Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        //private int regularizerIndex = 1;
        
        //@Parameter( names = "--regularize", description = "Use regularizer", required = false )
        //private boolean regularize = false;        
                        
        //@Parameter( names = "--lambda", description = "Regularizer lambda", required = false )
        //private float lambda = 0.1f;        
                        
        @Parameter( names = "--maxEpsilon", description = "Max epsilon", required = false )
        private float maxEpsilon = 200.0f;
                        
        @Parameter( names = "--minInlierRatio", description = "Min inlier ratio", required = false )
        private float minInlierRatio = 0.0f;
                        
        @Parameter( names = "--minNumInliers", description = "Min number of inliers", required = false )
        private int minNumInliers = 12;
                                
        //@Parameter( names = "--pointMatchScale", description = "Point match scale factor", required = false )
        //private double pointMatchScale = 1.0;
        
        @Parameter( names = "--rejectIdentity", description = "Reject identity transform solutions (ignore constant background)", required = false )
        private boolean rejectIdentity = false;
        
        @Parameter( names = "--identityTolerance", description = "Identity transform rejection tolerance", required = false )
        private float identityTolerance = 5.0f;
        
        //@Parameter( names = "--multipleHypotheses", description = "Return multiple hypotheses", required = false )
        //private boolean multipleHypotheses = false;
        
        @Parameter( names = "--maxNumFailures", description = "Max number of consecutive layer-to-layer match failures", required = false )
        private int maxNumFailures = 3;
        
        //@Parameter( names = "--maxIterationsOptimize", description = "Max iterations for optimization", required = false )
        //private int maxIterationsOptimize = 1000;
        
        //@Parameter( names = "--maxPlateauwidthOptimize", description = "Max plateau width for optimization", required = false )
        //private int maxPlateauwidthOptimize = 200;

        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
	}
	
	private FilterRansac() {}
	
	private static CoordinateTransform filterRansacTwoLayers( 
			final Params params, 
			final List< PointMatch > candidates,
			final String layerNameA,
			final String layerNameB )
	{
		CoordinateTransform resultModel = null;
		int attempt;
		
		// Try finding RANSAC (Random Sample Consensus) for params.maxNumFailures time before giving up
		for ( attempt = 0; attempt < params.maxNumFailures && resultModel == null; attempt++ )
		{
			CoordinateTransform model;
			//AbstractModel< ? > model;
			switch ( params.modelIndex )
			{
			case 0:
				model = new TranslationModel2D();
				break;
			case 1:
				model = new RigidModel2D();
				break;
			case 2:
				model = new SimilarityModel2D();
				break;
			case 3:
				model = new AffineModel2D();
				break;
			case 4:
				model = new HomographyModel2D();
				break;
			default:
				System.err.println( "Default model index could not be found, aborting Filter-Ransac." );
				throw new RuntimeException( "Default model index could not be found, aborting Filter-Ransac." );
			}
			
			final ArrayList< PointMatch > inliers = new ArrayList< PointMatch >();
			
			boolean modelFound;
			boolean again = false;
			try
			{
				do
				{
					again = false;
					modelFound = ( (AbstractModel< ? >) model).filterRansac(
								candidates,
								inliers,
								1000,
								params.maxEpsilon * params.layerScale,
								params.minInlierRatio,
								params.minNumInliers,
								3 );
					if ( modelFound && params.rejectIdentity )
					{
						final ArrayList< Point > points = new ArrayList< Point >();
						PointMatch.sourcePoints( inliers, points );
						if ( Transforms.isIdentity( model, points, params.identityTolerance *  params.layerScale ) )
						{
							System.out.println( "Identity transform for " + inliers.size() + " matches rejected." );
							//IJ.log( "Identity transform for " + inliers.size() + " matches rejected." );
							candidates.removeAll( inliers );
							inliers.clear();
							again = true;
						}
					}
				}
				while ( again );
			}
			catch ( final NotEnoughDataPointsException e )
			{
				modelFound = false;
			}
			
			if ( modelFound )
			{
				System.out.println( layerNameB + " -> " + layerNameA + ": " + inliers.size() + " corresponding features with an average displacement of " + ( PointMatch.meanDistance( inliers ) / params.layerScale ) + "px identified." );
				System.out.println( "Estimated transformation model: " + model );
				//models.set( ti, new Triple< Integer, Integer, AbstractModel< ? > >( sliceA, sliceB, model ) );
				resultModel = model;
			}
			else
			{
				System.out.println( layerNameB + " -> " + layerNameA + ": no correspondences found." );
				resultModel = null;
			}

		}

		if ( ( attempt == params.maxNumFailures )  && ( resultModel == null ) )
		{
			System.out.println( "Could not find a RANSAC model after " + params.maxNumFailures + " iterations!" );
			return null;
		}

		return resultModel;
	}
	
	private static int readLayer( String tileSpecUrl )
	{
		final URL url;
		final TileSpec[] tileSpecs;
		
		try
		{
			final Gson gson = new Gson();
			url = new URL( tileSpecUrl );
			tileSpecs = gson.fromJson( new InputStreamReader( url.openStream() ), TileSpec[].class );
		}
		catch ( final MalformedURLException e )
		{
			System.err.println( "URL malformed." );
			e.printStackTrace( System.err );
			throw new RuntimeException( e );
		}
		catch ( final JsonSyntaxException e )
		{
			System.err.println( "JSON syntax malformed." );
			e.printStackTrace( System.err );
			throw new RuntimeException( e );
		}
		catch ( final Exception e )
		{
			e.printStackTrace( System.err );
			throw new RuntimeException( e );
		}
		
		// Read layer
		return tileSpecs[0].layer;
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
		
		final CorrespondenceSpec[] corr_data;
		try
		{
			final Gson gson = new Gson();
			URL url = new URL( params.inputfile );
			corr_data = gson.fromJson( new InputStreamReader( url.openStream() ), CorrespondenceSpec[].class );
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

		// The mipmap level to work on
		// TODO: Should be a parameter from the user,
		//       and decide whether or not to create the mipmaps if they are missing
		//int mipmapLevel = 0;
		

		// Initialize threads pool
		final ExecutorService threadsPool = Executors.newFixedThreadPool( params.numThreads );
		final CoordinateTransform[] models = new CoordinateTransform[ corr_data.length ];
		final Future< ? >[] futures = new Future< ? > [ corr_data.length ];
		
		// parse all correspondence points
		for ( int i = 0; i < corr_data.length; i++ )
		{
			final int corrDataIndex = i;
			final CorrespondenceSpec corr = corr_data[ i ];
			final boolean equalUrl1 = params.compareUrl.equalsIgnoreCase( corr.url1 );
			final boolean equalUrl2 = params.compareUrl.equalsIgnoreCase( corr.url2 );
			if ( ( !equalUrl1 ) && ( !equalUrl2 ) )
			{
				System.out.println( "Error when parsing correspondece " + i + ", url " + params.compareUrl + " not found." );
			}
			else
			{
				final String otherUrl = ( ( equalUrl1 ) ? corr.url2 : corr.url1 );
				final int layer1 = readLayer( params.compareUrl );
				final int layer2 = readLayer( otherUrl );

				final String layerNameA = "Layer " + layer1;
				final String layerNameB = "Layer " + layer2;

				Future< ? > future = threadsPool.submit( new Runnable() {
					
					@Override
					public void run() {
						CoordinateTransform model = filterRansacTwoLayers( params, corr.correspondencePointPairs, layerNameA, layerNameB );
						models[ corrDataIndex ] = model;
					}
				});
				futures[ corrDataIndex ] = future;
			}
		}
		
		
		for ( Future< ? > future : futures )
		{
			try {
				future.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		threadsPool.shutdown();

		// Save all models to disk
		final List< ModelSpec > modelSpecs = new ArrayList<ModelSpec>();
		for ( int i = 0; i < models.length; i++ )
		{
			CoordinateTransform model = models[i];
			if ( model != null )
			{
				final ModelSpec ms = new ModelSpec( corr_data[i].url1, corr_data[i].url2, Transform.createTransform( model ) );
				modelSpecs.add( ms );
			}
		}
		
		System.out.println( "Exporting models.");
		
		try {
			Writer writer = new FileWriter( params.targetPath );
	        //Gson gson = new GsonBuilder().create();
	        Gson gson = new GsonBuilder().setPrettyPrinting().create();
	        gson.toJson( modelSpecs, writer );
	        writer.close();
	    }
		catch ( final IOException e )
		{
			System.err.println( "Error writing JSON file: " + params.targetPath );
			e.printStackTrace( System.err );
		}

		System.out.println( "Done." );

	}
	
}

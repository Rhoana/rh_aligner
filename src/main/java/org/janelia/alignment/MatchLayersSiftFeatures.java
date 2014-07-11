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

import java.io.FileReader;
import java.io.IOException;
import java.io.Writer;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.janelia.alignment.FeatureSpec.ImageAndFeatures;

import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.FloatArray2DSIFT;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

public class MatchLayersSiftFeatures
{
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--tilespec1", description = "The url of the first layer json file", required = true )
        private String tilespec1;

        @Parameter( names = "--tilespec2", description = "The url of the second layer json file", required = true )
        private String tilespec2;
		
        @Parameter( names = "--featurefile1", description = "Feature file of the first layer", required = true )
        private String featurefile1;

        @Parameter( names = "--featurefile2", description = "Feature file of the second layer", required = true )
        private String featurefile2;

        @Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;

        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

        @Parameter( names = "--rod", description = "ROD", required = false )
        public float rod = 0.92f;

        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        private float layerScale = 0.1f;
        

	}

	private MatchLayersSiftFeatures() {}


	/**
	 * Identify corresponding features
	 * 
	 * @param fs1 feature collection from set 1
	 * @param fs2 feature collection from set 2
	 * @param rod Ratio of distances (closest/next closest match)
	 * @param number of threads
	 * 
	 * @return matches
	 */
	public static Vector< PointMatch > createMatches(
			final List< Feature > fs1,
			final List< Feature > fs2,
			final float rod,
			final int threadsNum )
	{
		final Vector< PointMatch > matches = new Vector< PointMatch >();
		
		@SuppressWarnings("unchecked")
		final List< Feature >[] fs1Sublists = new List[ threadsNum ];
		
		// divide the first feature list between the threads
		final int[] fs1Indices = new int[ threadsNum + 1 ];
		final int fsPerThread = fs1.size() / threadsNum;
		fs1Indices[ 0 ] = 0;
		for ( int i = 1; i < threadsNum; i++ )
		{
			fs1Indices[ i ] = fs1Indices[ i - 1 ] + fsPerThread;
			fs1Sublists[ i - 1 ] = fs1.subList( fs1Indices[ i - 1 ], fs1Indices[ i ] );
		}
		fs1Indices[ threadsNum ] = fs1.size();
		fs1Sublists[ threadsNum - 1 ] = fs1.subList( fs1Indices[ threadsNum - 1 ], fs1Indices[ threadsNum ] );

		final Future< ? >[] futures = new Future< ? >[ threadsNum ];
		ExecutorService threadsPool = Executors.newFixedThreadPool( threadsNum );	
		
		for ( int t = 0; t < threadsNum ; t++ )
		{
			final int threadIndex = t;
			futures[ t ] = threadsPool.submit( new Runnable() {

				@Override
				public void run() {
					final List< PointMatch > threadMatches = FloatArray2DSIFT.createMatches(
							fs1Sublists[ threadIndex ], fs2, rod );
					synchronized( matches )
					{
						matches.addAll( threadMatches );
					}
				}
			});			
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

		// now remove ambiguous matches
		for ( int i = 0; i < matches.size(); )
		{
			boolean amb = false;
			final PointMatch m = matches.get( i );
			final float[] m_p2 = m.getP2().getL(); 
			for ( int j = i + 1; j < matches.size(); )
			{
				final PointMatch n = matches.get( j );
				final float[] n_p2 = n.getP2().getL(); 
				if ( m_p2[ 0 ] == n_p2[ 0 ] && m_p2[ 1 ] == n_p2[ 1 ] )
				{
					amb = true;
					matches.removeElementAt( j );
				}
				else ++j;
			}
			if ( amb )
				matches.removeElementAt( i );
			else ++i;
		}
		return matches;
	}

	
	private static List< Feature > getAllFeatures( FeatureSpec[] featureSpecs, int mipmapLevel )
	{
		final List< Feature > featuresList = new ArrayList<Feature>();
		
		for ( FeatureSpec fs : featureSpecs )
		{
			final ImageAndFeatures iaf = fs.getMipmapImageAndFeatures( mipmapLevel );
			featuresList.addAll( iaf.featureList );
		}
		
		return featuresList;
	}

	private static double getScale( FeatureSpec[] featureSpecs, int mipmapLevel )
	{
		double res = -1.0;
		
		for ( FeatureSpec fs : featureSpecs )
		{
			final ImageAndFeatures iaf = fs.getMipmapImageAndFeatures( mipmapLevel );
			if ( res == -1.0 )
				res = iaf.scale;
			else
				if ( res != iaf.scale )
					throw new RuntimeException( "Error: different scales of sift features in a single file" );
		}
		
		return res;
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

		/* open featurespec */
		final FeatureSpec[] featureSpecs1;
		final FeatureSpec[] featureSpecs2;
		try
		{
			final Gson gson = new Gson();
			System.out.println( "Reading features from " + params.featurefile1 );
			featureSpecs1 = gson.fromJson( new FileReader( params.featurefile1.replace("file://", "").replace("file:/", "") ), FeatureSpec[].class );
			System.out.println( "Reading features from " + params.featurefile2 );
			featureSpecs2 = gson.fromJson( new FileReader( params.featurefile2.replace("file://", "").replace("file:/", "") ), FeatureSpec[].class );
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

		List< CorrespondenceSpec > corr_data = new ArrayList< CorrespondenceSpec >();

		final List< Feature > fs1 = getAllFeatures( featureSpecs1, mipmapLevel );
		final List< Feature > fs2 = getAllFeatures( featureSpecs2, mipmapLevel );

		System.out.println( "Searching for matching candidates" );
		//			final List< PointMatch > candidates = new ArrayList< PointMatch >();
//			FeatureTransform.matchFeatures( fs1, fs2, candidates, params.rod );
		final List< PointMatch > candidates = FloatArray2DSIFT.createMatches( fs2, fs1, params.rod );
		// To be consistent with the original matches calculation, we match fs2 -> fs1
		//final List< PointMatch > candidates = createMatches( fs2, fs1, params.rod, params.numThreads );

		System.out.println( "Found " + candidates.size() + " matching candidates, scaling..." );

		final double pointMatchScale1 = params.layerScale / getScale( featureSpecs2, mipmapLevel );
		final double pointMatchScale2 = params.layerScale / getScale( featureSpecs1, mipmapLevel );

		/* scale the candidates */
		for ( final PointMatch pm : candidates )
		{
			final Point p1 = pm.getP1();
			final Point p2 = pm.getP2();
			final float[] l1 = p1.getL();
			final float[] w1 = p1.getW();
			final float[] l2 = p2.getL();
			final float[] w2 = p2.getW();
			
			l1[ 0 ] *= pointMatchScale1;
			l1[ 1 ] *= pointMatchScale1;
			w1[ 0 ] *= pointMatchScale1;
			w1[ 1 ] *= pointMatchScale1;
			l2[ 0 ] *= pointMatchScale2;
			l2[ 1 ] *= pointMatchScale2;
			w2[ 0 ] *= pointMatchScale2;
			w2[ 1 ] *= pointMatchScale2;
			
			System.out.println( "* Candidate: L(" + l1[0] + "," + l1[1] + ") -> L(" + l2[0] + "," + l2[1] + ")" );
		}
		

		corr_data.add(new CorrespondenceSpec(mipmapLevel,
				params.tilespec1,
				params.tilespec2,
				candidates));

		if (corr_data.size() > 0) {
			try {
				Writer writer = new FileWriter(params.targetPath);
				//Gson gson = new GsonBuilder().create();
				Gson gson = new GsonBuilder().setPrettyPrinting().create();
				gson.toJson(corr_data, writer);
				writer.close();
			}
			catch ( final IOException e )
			{
				System.err.println( "Error writing JSON file: " + params.targetPath );
				e.printStackTrace( System.err );
			}
		}
		
		System.out.println( "Done." );
	}
}

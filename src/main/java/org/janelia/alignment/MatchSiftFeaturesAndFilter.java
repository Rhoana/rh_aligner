package org.janelia.alignment;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import mpicbg.ij.FeatureTransform;
import mpicbg.imagefeatures.Feature;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.Model;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.Transforms;
import mpicbg.models.TranslationModel2D;

import org.janelia.alignment.FeatureSpec.ImageAndFeatures;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;
import com.sun.tools.corba.se.idl.toJavaPortable.Util;

public class MatchSiftFeaturesAndFilter {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--tilespecfile", description = "TileSpec file", required = true )
        private String tilespecfile;

        @Parameter( names = "--featurefile", description = "Feature file", required = true )
        private String featurefile;

        @Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;

        @Parameter( names = "--indices", description = "Pair of indices within feature file, comma separated (each pair is separated by a colon)", required = true )
        public List<String> indices = new ArrayList<String>();

        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

        @Parameter( names = "--rod", description = "ROD", required = false )
        public float rod = 0.5f;
        
        @Parameter( names = "--tileScale", description = "Tile scale (to search for matches)", required = false )
        private float tileScale = 1.0f;
        
        @Parameter( names = "--modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int modelIndex = 1;
        
        @Parameter( names = "--maxEpsilon", description = "Max epsilon", required = false )
        private float maxEpsilon = 200.0f;

        @Parameter( names = "--minInlierRatio", description = "Min inlier ratio", required = false )
        private float minInlierRatio = 0.0f;

        @Parameter( names = "--minNumInliers", description = "Min number of inliers", required = false )
        private int minNumInliers = 12;
        
        @Parameter( names = "--rejectIdentity", description = "Reject identity transform solutions (ignore constant background)", required = false )
        private boolean rejectIdentity = false;

        @Parameter( names = "--identityTolerance", description = "Identity transform rejection tolerance", required = false )
        private float identityTolerance = 5.0f;
	}

	private MatchSiftFeaturesAndFilter() { }
	
	final static public boolean findModel(
			final Model< ? > model,
			final List< PointMatch > candidates,
			final Collection< PointMatch > inliers,
			final float maxEpsilon,
			final float minInlierRatio,
			final int minNumInliers,
			final boolean rejectIdentity,
			final float identityTolerance )
	{
		boolean modelFound;
		boolean again = false;
		try
		{
			do
			{
				again = false;
				modelFound = model.filterRansac(
						candidates,
						inliers,
						1000,
						maxEpsilon,
						minInlierRatio,
						minNumInliers,
						3 );
				if ( modelFound && rejectIdentity )
				{
					final ArrayList< Point > points = new ArrayList< Point >();
					PointMatch.sourcePoints( inliers, points );
					if ( Transforms.isIdentity( model, points, identityTolerance ) )
					{
						System.out.println( "Identity transform for " + inliers.size() + " matches rejected." );
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

		return modelFound;
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

		/* open tilespec */
		final TileSpec[] tileSpecs;
		try
		{
			final URL url;
			final Gson gson = new Gson();
			url = new URL( params.tilespecfile );
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

		/* open featurespec */
		final FeatureSpec[] featureSpecs;
		try
		{
			final Gson gson = new Gson();
			featureSpecs = gson.fromJson( new FileReader( params.featurefile.replace("file://", "").replace("file:/", "") ), FeatureSpec[].class );
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

		for (String idx_pair : params.indices) {
			String[] vals = idx_pair.split(":");
			if (vals.length != 2)
				throw new IllegalArgumentException("Index pair not in correct format:" + idx_pair);
			int idx1 = Integer.parseInt(vals[0]);
			int idx2 = Integer.parseInt(vals[1]);
			
			final ImageAndFeatures iaf1 = featureSpecs[idx1].getMipmapImageAndFeatures( mipmapLevel );
			final ImageAndFeatures iaf2 = featureSpecs[idx2].getMipmapImageAndFeatures( mipmapLevel );

			final List< Feature > fs1 = iaf1.featureList;
			final List< Feature > fs2 = iaf2.featureList;


			final List< PointMatch > candidates = new ArrayList< PointMatch >();
			//FeatureTransform.matchFeatures( fs2, fs1, candidates, params.rod );
			FeatureTransform.matchFeatures( fs1, fs2, candidates, params.rod );

			
			/* scale the candidates */
			final CoordinateTransformList< CoordinateTransform > ctl1 = tileSpecs[idx1].createTransformList();
			final CoordinateTransformList< CoordinateTransform > ctl2 = tileSpecs[idx2].createTransformList();
			for ( final PointMatch pm : candidates )
			{
				final Point p1 = pm.getP1();
				final Point p2 = pm.getP2();
								
				final float[] l1 = p1.getL();
				final float[] w1 = p1.getW();
				final float[] l2 = p2.getL();
				final float[] w2 = p2.getW();

				// Apply original transformations on the candidates
				l1[ 0 ] *= 1.0 / iaf1.scale;
				l1[ 1 ] *= 1.0 / iaf1.scale;
				w1[ 0 ] *= 1.0 / iaf1.scale;
				w1[ 1 ] *= 1.0 / iaf1.scale;
				ctl1.applyInPlace( w1 );
				l2[ 0 ] *= 1.0 / iaf2.scale;
				l2[ 1 ] *= 1.0 / iaf2.scale;
				w2[ 0 ] *= 1.0 / iaf2.scale;
				w2[ 1 ] *= 1.0 / iaf2.scale;
				ctl2.applyInPlace( w2 );

				System.out.println( "* Candidate: L(" + l1[0] + "," + l1[1] + ") -> L(" + l2[0] + "," + l2[1] + ")" );

				// Scale the candidates by some parameter scale
				l1[ 0 ] *= params.tileScale;
				l1[ 1 ] *= params.tileScale;
				w1[ 0 ] *= params.tileScale;
				w1[ 1 ] *= params.tileScale;
				l2[ 0 ] *= params.tileScale;
				l2[ 1 ] *= params.tileScale;
				w2[ 0 ] *= params.tileScale;
				w2[ 1 ] *= params.tileScale;
			}

			/* Filter the matches */
            /* find the model */
			
			final Model< ? > model = Utils.createModel( params.modelIndex );
			
//			final AbstractAffineModel2D< ? > model;
//			switch ( p.expectedModelIndex )
//			{
//			case 0:
//				model = new TranslationModel2D();
//				break;
//			case 1:
//				model = new RigidModel2D();
//				break;
//			case 2:
//				model = new SimilarityModel2D();
//				break;
//			case 3:
//				model = new AffineModel2D();
//				break;
//			default:
//				return;
//			}
//

			//List< PointMatch > inliers = candidates;
			
			List< PointMatch > inliers = new ArrayList< PointMatch >();
			
			final boolean modelFound = findModel(
					model,
					candidates,
					inliers,
					params.maxEpsilon,
					params.minInlierRatio,
					params.minNumInliers,
					params.rejectIdentity,
					params.identityTolerance );


			if ( modelFound )
				System.out.println( "Model found for tiles at indices: " + idx1 + " and " + idx2 + ":\n  correspondences  " + inliers.size() + " of " + candidates.size() + "\n  average residual error  " + model.getCost() + " px" );
			else
				System.out.println( "No model found for tiles at indices: " + idx1 + " and " + idx2 + "\":\n  correspondence candidates  " + candidates.size() );
			
			
			corr_data.add(new CorrespondenceSpec(mipmapLevel,
					iaf1.imageUrl,
					iaf2.imageUrl,
					inliers));
		}

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
	}

}

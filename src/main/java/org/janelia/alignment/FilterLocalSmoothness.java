package org.janelia.alignment;

import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import mpicbg.models.AbstractModel;
import mpicbg.models.InvertibleCoordinateTransform;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

public class FilterLocalSmoothness {
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
		private final boolean help = false;

		@Parameter( names = "--inputfile", description = "Correspondence json file", variableArity = true, required = true )
		public String corrFile;

        @Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;

		@Parameter( names = "--layerScale", description = "Layer scale", required = false )
		public float layerScale = 0.1f;

		@Parameter( names = "--localModelIndex", description = "localModelIndex", required = false )
		public int localModelIndex = 1;
		// 0 = "Translation", 1 = "Rigid", 2 = "Similarity", 3 = "Affine"

		@Parameter( names = "--localRegionSigma", description = "localRegionSigma", required = false )
		public float localRegionSigma = 200f;

		@Parameter( names = "--maxLocalEpsilon", description = "maxLocalEpsilon", required = false )
		public float maxLocalEpsilon = 12f;

		@Parameter( names = "--maxLocalTrust", description = "maxLocalTrust", required = false )
		public int maxLocalTrust = 3;

		@Parameter( names = "--cropNSigma", description = "cropNSigma", required = false )
		public int cropNSigma = 6;

        @Parameter( names = "--useLegacyFilter", description = "Use legacy filter", required = false )
        private boolean useLegacyFilter = false;


	}

	/**
	 * 
	 * @param pm PointMatches
	 * @param min x = min[0], y = min[1]
	 * @param max x = max[0], y = max[1]
	 */
	private static void calculateBoundingBox(
			final List< PointMatch > pm,
			final float[] min,
			final float[] max )
	{
		final float[] first = pm.get( 0 ).getP1().getW();
		min[ 0 ] = first[ 0 ];
		min[ 1 ] = first[ 1 ];
		max[ 0 ] = first[ 0 ];
		max[ 1 ] = first[ 1 ];

		for ( final PointMatch p : pm )
		{
			final float[] t = p.getP1().getW();
			if ( t[ 0 ] < min[ 0 ] ) min[ 0 ] = t[ 0 ];
			else if ( t[ 0 ] > max[ 0 ] ) max[ 0 ] = t[ 0 ];
			if ( t[ 1 ] < min[ 1 ] ) min[ 1 ] = t[ 1 ];
			else if ( t[ 1 ] > max[ 1 ] ) max[ 1 ] = t[ 1 ];
		}
	}

	


	public static boolean croppedLocalSmoothnessFilter(
			final AbstractModel< ? > localSmoothnessFilterModel,
			final List< PointMatch > candidates,
			final List< PointMatch > inliers,
			final double sigma,
			final double maxEpsilon,
			final double maxTrust,
			final int cropNSigma)
	{
		if (candidates.size() < localSmoothnessFilterModel.getMinNumMatches())
			return false;

		final double var2 = 2 * sigma * sigma;

		/* unshift an extra weight into candidates */
		for ( final PointMatch match : candidates )
			match.unshiftWeight( 1.0f );

		/* initialize inliers */
		if ( inliers != candidates )
		{
			inliers.clear();
			inliers.addAll( candidates );
		}

		/* determine inlier bounding box */
		final float[] pmin = new float[ 2 ];
		final float[] pmax = new float[ 2 ];
		calculateBoundingBox( inliers, pmin, pmax );

		final float gridD = (float)sigma * cropNSigma / 2;
		final int gridW = Math.max(1, (int)Math.round((pmax[0] - pmin[0]) / gridD));
		final int gridH = Math.max(1, (int)Math.round((pmax[1] - pmin[1]) / gridD));

		/* evenly distribute the grid to make sure the last row / col is not too small */
		final float gridDW = (pmax[0] - pmin[0]) / gridW;
		final float gridDH = (pmax[1] - pmin[1]) / gridH;
		System.out.println( "Filter sigma: " + sigma + ", gridD: " + gridD + " (after scale)" );
		System.out.println( "grid " + gridW + "x" + gridH + ", " + gridDW + "x" + gridDH );

		/* split into n-sigma / 2 wide bins to restrict search space */
		//ArrayList<PointMatch>[][] grid = new ArrayList<PointMatch>[gridW][gridH];
		ArrayList<ArrayList<ArrayList<PointMatch>>> grid = new ArrayList<ArrayList<ArrayList<PointMatch>>>(gridW);
		for ( int i = 0; i < gridW; ++i )
		{
			grid.add(new ArrayList<ArrayList<PointMatch>>(gridH));
			for ( int j = 0; j < gridH; ++j )
			{
				grid.get(i).add(new ArrayList<PointMatch>());
			}
		}

		for ( final PointMatch match : candidates )
		{
			int gridi = (int)Math.floor((match.getP1().getW()[0] - pmin[0]) / gridDW);
			int gridj = (int)Math.floor((match.getP1().getW()[1] - pmin[1]) / gridDH);
			if (gridi >= gridW)
				gridi = gridW - 1;
			if (gridj >= gridH)
				gridj = gridH - 1;
			//System.out.println(match.getP1().getW()[0] + "x" + match.getP1().getW()[1] + " -> " + gridi + "x" + gridj);
			grid.get(gridi).get(gridj).add(match);
		}

		//		for ( int i = 0; i < gridW; ++i )
		//		{
		//			for ( int j = 0; j < gridH; ++j )
		//			{
		//				System.out.println("grid " + i + "," + j + " size=" + grid.get(i).get(j).size());
		//			}
		//		}

		boolean hasChanged = false;

		int p = 0;
		do
		{
			System.out.println( "Smoothness filter pass " + String.format( "%2d", ++p ));
			/*
			// For debug:
			int sum = 0;
			for (int centrali = 0; centrali < gridW; ++centrali)
			{
				for (int centralj = 0; centralj < gridH; ++centralj)
				{
					sum += grid.get(centrali).get(centralj).size();
				}
			}
			System.out.println( "Iteration candidates#: " + sum );
			*/
			
			hasChanged = false;

			final ArrayList< PointMatch > toBeRemoved = new ArrayList< PointMatch >();
			final ArrayList< PointMatch > localInliers = new ArrayList< PointMatch >();
			final ArrayList< PointMatch > localCandidates = new ArrayList< PointMatch >();

			//			final int i = 0;

			for (int centrali = 0; centrali < gridW; ++centrali)
			{
				for (int centralj = 0; centralj < gridH; ++centralj)
				{

					//System.out.println("Block " + centrali + " " + centralj);

					ArrayList<PointMatch> centralInliers = grid.get(centrali).get(centralj);

					for ( final PointMatch candidate : centralInliers )
					{
						//System.out.println( "loop1" );
						localCandidates.clear();

						/* calculate weights by square distance to reference in local space */
						for (int offseti = -1; offseti <= 1; ++offseti)
						{
							int outeri = centrali + offseti;
							if (outeri < 0 || outeri >= gridW)
								continue;

							for (int offsetj = -1; offsetj <= 1; ++offsetj)
							{
								int outerj = centralj + offsetj;
								if (outerj < 0 || outerj >= gridH)
									continue;

								//System.out.println("  Outer Block " + outeri + " " + outerj);
								ArrayList<PointMatch> outerInliers = grid.get(outeri).get(outerj);
								
								//System.out.println( "outerInliers#: " + outerInliers.size() );

								for ( final PointMatch match : outerInliers )
								{
									final float dist = Point.localDistance( candidate.getP1(), match.getP1() );
									if (dist > gridD)
										continue;
									final float w = ( float )Math.exp( -(dist*dist) / var2 );
									match.setWeight( 0, w );
									localCandidates.add(match);
								}
							}
						}

						candidate.setWeight( 0, 0 );

						boolean filteredLocalModelFound;
						try
						{
							filteredLocalModelFound = localSmoothnessFilterModel.filter( localCandidates, localInliers, ( float )maxTrust );
						}
						catch ( final NotEnoughDataPointsException e )
						{
							filteredLocalModelFound = false;
						}

						if ( !filteredLocalModelFound )
						{
							hasChanged = true;
							toBeRemoved.add( candidate );
						}
						else
						{
	
							candidate.apply( localSmoothnessFilterModel );
							final double candidateDistance = Point.distance( candidate.getP1(), candidate.getP2() );
	
							//
							// TESTING - ignore trust region - just use maxEpsilon
							// 
							if ( candidateDistance > maxEpsilon )
							{
								hasChanged = true;
								toBeRemoved.add( candidate );
							}
							else
							{
								PointMatch.apply( localCandidates, localSmoothnessFilterModel );
	
								/* weighed mean Euclidean distances */
								double meanDistance = 0, ws = 0;
								for ( final PointMatch match : inliers )
								{
									final float w = match.getWeight();
									ws += w;
									meanDistance += Point.distance( match.getP1(), match.getP2() ) * w;
								}
								meanDistance /= ws;
	
								if ( candidateDistance > maxTrust * meanDistance )
								{
									hasChanged = true;
									toBeRemoved.add( candidate );
								}
	
								PointMatch.apply( localCandidates, ((InvertibleCoordinateTransform)localSmoothnessFilterModel).createInverse() );
							}
						}
					}

//					/* reset weights for next grid point - only necessary if all inliers are being considered */
//					
//					for (int offseti = -1; offseti <= 1; ++offseti)
//					{
//						int outeri = centrali + offseti;
//						if (outeri < 0 || outeri >= gridW)
//							continue;
//						
//						for (int offsetj = -1; offsetj <= 1; ++offsetj)
//						{
//							int outerj = centralj + offsetj;
//							if (outerj < 0 || outerj >= gridH)
//								continue;
//							
//							//System.out.println("  Outer Block " + outeri + " " + outerj);
//							ArrayList<PointMatch> outerInliers = grid.get(outeri).get(outerj);
//							
//							for ( final PointMatch match : outerInliers )
//							{
//								match.setWeight( 0, 0 );
//							}
//						}
//					}

				}
			}
			inliers.removeAll( toBeRemoved );
			for ( int i = 0; i < gridW; ++i )
			{
				for ( int j = 0; j < gridH; ++j )
				{
					grid.get(i).get(j).removeAll( toBeRemoved );
				}
			}

			//			System.out.println();
		}
		while ( hasChanged );

		/* clean up extra weight from candidates */
		for ( final PointMatch match : candidates )
			match.shiftWeight();

		return inliers.size() >= localSmoothnessFilterModel.getMinNumMatches();
	}	



	public static boolean localSmoothnessFilterTest(
			final AbstractModel< ? > localSmoothnessFilterModel,
			final Collection< PointMatch > candidates,
			final Collection< PointMatch > inliers,
			final double sigma,
			final double maxEpsilon,
			final double maxTrust)
	{
		final double var2 = 2 * sigma * sigma;

		/* unshift an extra weight into candidates */
		for ( final PointMatch match : candidates )
			match.unshiftWeight( 1.0f );

		/* initialize inliers */
		if ( inliers != candidates )
		{
			inliers.clear();
			inliers.addAll( candidates );
		}

		boolean hasChanged = false;

		int p = 0;
		System.out.print( "Smoothness filter pass  1:   0%" );
		do
		{
			System.out.print( ( char )13 + "Smoothness filter pass " + String.format( "%2d", ++p ) + ":   0%" + ", Iteration candidates#: " + candidates.size() );
			hasChanged = false;

			final ArrayList< PointMatch > toBeRemoved = new ArrayList< PointMatch >();
			final ArrayList< PointMatch > localInliers = new ArrayList< PointMatch >();

			//			final int i = 0;

			for ( final PointMatch candidate : inliers )
			{
				//				System.out.print( ( char )13 + "Smoothness filter pass " + String.format( "%2d", p ) + ": " + String.format( "%3d", ( ++i * 100 / inliers.size() ) ) + "%" );

				/* calculate weights by square distance to reference in local space */
				for ( final PointMatch match : inliers )
				{
					final float w = ( float )Math.exp( -Point.squareLocalDistance( candidate.getP1(), match.getP1() ) / var2 );
					match.setWeight( 0, w );
				}

				candidate.setWeight( 0, 0 );

				boolean filteredLocalModelFound;
				try
				{
					filteredLocalModelFound = localSmoothnessFilterModel.filter( candidates, localInliers, ( float )maxTrust );
				}
				catch ( final NotEnoughDataPointsException e )
				{
					filteredLocalModelFound = false;
				}

				if ( !filteredLocalModelFound )
				{
					/* clean up extra weight from candidates */
					for ( final PointMatch match : candidates )
						match.shiftWeight();

					/* no inliers */
					inliers.clear();

					return false;
				}

				candidate.apply( localSmoothnessFilterModel );
				final double candidateDistance = Point.distance( candidate.getP1(), candidate.getP2() );
				if ( candidateDistance <= maxEpsilon )
				{
					PointMatch.apply( inliers, localSmoothnessFilterModel );

					/* weighed mean Euclidean distances */
					double meanDistance = 0, ws = 0;
					for ( final PointMatch match : inliers )
					{
						final float w = match.getWeight();
						ws += w;
						meanDistance += Point.distance( match.getP1(), match.getP2() ) * w;
					}
					meanDistance /= ws;

					if ( candidateDistance > maxTrust * meanDistance )
					{
						hasChanged = true;
						toBeRemoved.add( candidate );
					}
				}
				else
				{
					hasChanged = true;
					toBeRemoved.add( candidate );
				}
			}
			inliers.removeAll( toBeRemoved );
			//			System.out.println();
		}
		while ( hasChanged );

		/* clean up extra weight from candidates */
		for ( final PointMatch match : candidates )
			match.shiftWeight();

		return inliers.size() >= localSmoothnessFilterModel.getMinNumMatches();
	}
	
	private static void filter(
			final String fileUrl,
			final String outFile,
			final Params params )
	{
		// Open and parse the json file
		final CorrespondenceSpec[] corr_data;
		try
		{
			final Gson gson = new Gson();
			URL url = new URL( fileUrl );
			corr_data = gson.fromJson( new InputStreamReader( url.openStream() ), CorrespondenceSpec[].class );
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
		
		final AbstractModel< ? > localSmoothnessFilterModel = Utils.createModel( params.localModelIndex );
		final float localRegionSigma = params.layerScale * params.localRegionSigma;
		final float maxLocalEpsilon = params.layerScale * params.maxLocalEpsilon;
		
		for ( final CorrespondenceSpec corr : corr_data )
		{
			String layer1 = corr.url1;
			String layer2 = corr.url2;
			final List< PointMatch > pm12 = corr.correspondencePointPairs;


			System.out.println( layer1 + " > " + layer2 + ": found " + pm12.size() + " correspondence candidates." );
								
			long startTime = System.currentTimeMillis();
			if ( params.useLegacyFilter )
			{
				// This is the original version
				localSmoothnessFilterTest( localSmoothnessFilterModel, pm12, pm12, localRegionSigma, maxLocalEpsilon, params.maxLocalTrust );
				
				System.out.println( layer1 + " > " + layer2 + ": " + pm12.size() + " candidates passed (legacy) local smoothness filter." );
			}
			else
			{
				// This version performs the same operation, but restricts the search window to radius 2-sigma, effectively weighting all matches outside this area to zero.
				croppedLocalSmoothnessFilter( localSmoothnessFilterModel, pm12, pm12, localRegionSigma, maxLocalEpsilon, params.maxLocalTrust, params.cropNSigma );
				
				System.out.println( layer1 + " > " + layer2 + ": " + pm12.size() + " candidates passed (crop) local smoothness filter." );
			}
			long endTime = System.currentTimeMillis();
			System.out.println("Filter step took: " + ((endTime - startTime) / 1000.0) + " ms");
		}
		
		// corr_data was updated, just output it to the file
		long startTime = System.currentTimeMillis();
		try {
			Writer writer = new FileWriter( outFile );
	        //Gson gson = new GsonBuilder().create();
	        Gson gson = new GsonBuilder().setPrettyPrinting().create();
	        gson.toJson(corr_data, writer);
	        writer.close();
	    }
		catch ( final IOException e )
		{
			System.err.println( "Error writing JSON file: " + outFile );
			e.printStackTrace( System.err );
		}
		long endTime = System.currentTimeMillis();
		System.out.println("Writing output took: " + ((endTime - startTime) / 1000.0) + " ms");
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

		System.out.println( "Reading correspondence files" );

		filter( params.corrFile, params.targetPath, params );
		

		System.out.println( "Done." );
	}
}

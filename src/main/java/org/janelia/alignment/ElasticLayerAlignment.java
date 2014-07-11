package org.janelia.alignment;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ini.trakem2.utils.Filter;
//import ini.trakem2.utils.Utils;





import java.awt.Color;
import java.awt.Image;
import java.awt.Rectangle;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import javax.imageio.ImageIO;

import mpicbg.ij.SIFT;
import mpicbg.ij.blockmatching.BlockMatching;
import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.FloatArray2DSIFT;
import mpicbg.models.AbstractModel;
import mpicbg.models.AffineModel2D;
import mpicbg.models.ErrorStatistic;
import mpicbg.models.HomographyModel2D;
import mpicbg.models.IllDefinedDataPointsException;
import mpicbg.models.InvertibleCoordinateTransform;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.RigidModel2D;
import mpicbg.models.SimilarityModel2D;
import mpicbg.models.Spring;
import mpicbg.models.SpringMesh;
import mpicbg.models.Tile;
import mpicbg.models.TileConfiguration;
import mpicbg.models.Transforms;
import mpicbg.models.TranslationModel2D;
import mpicbg.models.Vertex;
import mpicbg.trakem2.align.Util;
import mpicbg.trakem2.transform.MovingLeastSquaresTransform;
import mpicbg.trakem2.transform.MovingLeastSquaresTransform2;
import mpicbg.trakem2.util.Triple;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

/**
 * A reconstruction of the entire 3D elastic alignment process
 * in a single file.
 * TODO: After this works well, we need to break it down to other functions and files
 * and remove this file.
 */
public class ElasticLayerAlignment {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

		@Parameter(description = "Files")
		private List<String> files = new ArrayList<String>();
		        
        @Parameter( names = "--modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int modelIndex = 1;
        /*
        @Parameter( names = "--meshWidth", description = "Mesh width (in pixels) for all images in this series.", required = true )
        public int meshWidth;
        
        @Parameter( names = "--meshHeight", description = "Mesh height (in pixels) for all images in this series.", required = true )
        public int meshHeight;
        */
        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        private float layerScale = 0.1f;
        
        @Parameter( names = "--resolutionSpringMesh", description = "resolutionSpringMesh", required = false )
        private int resolutionSpringMesh = 32;
        
//        @Parameter( names = "--springLengthSpringMesh", description = "springLengthSpringMesh", required = false )
//        private float springLengthSpringMesh = 100f;
		
        @Parameter( names = "--stiffnessSpringMesh", description = "stiffnessSpringMesh", required = false )
        private float stiffnessSpringMesh = 0.1f;
		
        @Parameter( names = "--dampSpringMesh", description = "dampSpringMesh", required = false )
        private float dampSpringMesh = 0.9f;
		
        @Parameter( names = "--maxStretchSpringMesh", description = "maxStretchSpringMesh", required = false )
        private float maxStretchSpringMesh = 2000.0f;
        
        @Parameter( names = "--maxEpsilon", description = "maxEpsilon", required = false )
        private float maxEpsilon = 200.0f;
        
        @Parameter( names = "--maxIterationsSpringMesh", description = "maxIterationsSpringMesh", required = false )
        private int maxIterationsSpringMesh = 1000;
        
        @Parameter( names = "--maxPlateauwidthSpringMesh", description = "maxPlateauwidthSpringMesh", required = false )
        private int maxPlateauwidthSpringMesh = 200;
        
//        @Parameter( names = "--resolutionOutput", description = "resolutionOutput", required = false )
//        private int resolutionOutput = 128;
        
        @Parameter( names = "--targetDir", description = "Directory where the output json tile spec files will be kept", required = true )
        public String targetDir;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
        @Parameter( names = "--maxNumNeighbors", description = "Number of neighboring sections to align", required = false )
        public int maxNumNeighbors = 1;

        @Parameter( names = "--useLegacyOptimizer", description = "Use legacy optimizer", required = false )
        private boolean useLegacyOptimizer = true;

        
        /* For SIFT features computation */
        @Parameter( names = "--maxOctaveSize", description = "Max image size", required = false )
        private int sift_maxOctaveSize = 1024;

        @Parameter( names = "--initialSigma", description = "Initial Gaussian blur sigma", required = false )
        private float sift_initialSigma = 1.6f;
        
        @Parameter( names = "--steps", description = "Steps per scale octave", required = false )
        private int sift_steps = 3;
        
        @Parameter( names = "--minOctaveSize", description = "Min image size", required = false )
        private int sift_minOctaveSize = 64;
        
        @Parameter( names = "--fdSize", description = "Feature descriptor size", required = false )
        private int sift_fdSize = 8;
        
        @Parameter( names = "--fdBins", description = "Feature descriptor bins", required = false )
        private int sift_fdBins = 8;

        /* For matching between sections */
        @Parameter( names = "--rod", description = "ROD", required = false )
        public float match_rod = 0.92f;

        /* For ransac sections */
        @Parameter( names = "--minInlierRatio", description = "Min inlier ratio", required = false )
        private float ransac_minInlierRatio = 0.0f;
                        
        @Parameter( names = "--minNumInliers", description = "Min number of inliers", required = false )
        private int ransac_minNumInliers = 12;
        
        @Parameter( names = "--rejectIdentity", description = "Reject identity transform solutions (ignore constant background)", required = false )
        private boolean ransac_rejectIdentity = false;
        
        @Parameter( names = "--identityTolerance", description = "Identity transform rejection tolerance", required = false )
        private float ransac_identityTolerance = 5.0f;

        @Parameter( names = "--maxNumFailures", description = "Max number of consecutive layer-to-layer match failures", required = false )
        private int ransac_maxNumFailures = 3;

        @Parameter( names = "--ransac_modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int ransac_modelIndex = 3;

        /* For Match by Max PMCC */
        @Parameter( names = "--searchRadius", description = "Search window radius", required = false )
        public int pmcc_searchRadius = 200;
        
        @Parameter( names = "--blockRadius", description = "Matching block radius", required = false )
        public int pmcc_blockRadius = 579;

        @Parameter( names = "--pmccResolution", description = "pmccResolution", required = false )
        private int pmcc_resolution = 16;

        @Parameter( names = "--localRegionSigma", description = "localRegionSigma", required = false )
        public float pmcc_localRegionSigma = 200f;
        
        @Parameter( names = "--maxLocalEpsilon", description = "maximal_local_displacement (absolute)", required = false )
        public float pmcc_maxLocalEpsilon = 12f;

        @Parameter( names = "--minR", description = "minR", required = false )
        public float pmcc_minR = 0.6f;
        
        @Parameter( names = "--maxCurvatureR", description = "maxCurvatureR", required = false )
        public float pmcc_maxCurvatureR = 10.0f;
        
        @Parameter( names = "--rodR", description = "rodR", required = false )
        public float pmcc_rodR = 0.9f;
        
        @Parameter( names = "--useLocalSmoothnessFilter", description = "useLocalSmoothnessFilter", required = false )
        //public boolean pmcc_useLocalSmoothnessFilter = true;
        public boolean pmcc_useLocalSmoothnessFilter = false;

        @Parameter( names = "--maxLocalTrust", description = "maximal_local_displacement (relative)", required = false )
        public int pmcc_maxLocalTrust = 3;

        @Parameter( names = "--pmmcModelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int pmcc_localModelIndex = 1;

	}

	private ElasticLayerAlignment() { }
	
	private static Params parseParams( String[] args )
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
        	jc.setProgramName( "java [-options] -cp render.jar org.janelia.alignment.RenderTile" );
        	jc.usage(); 
        	return null;
        }
		
		return params;
	}
	
	/*
	private static CorrespondenceSpec[] readCorrespondenceFile( String inputfile )
	{
		final CorrespondenceSpec[] corr_data;
		try
		{
			final Gson gson = new Gson();
			URL url = new URL( inputfile );
			corr_data = gson.fromJson( new InputStreamReader( url.openStream() ), CorrespondenceSpec[].class );
		}
		catch ( final MalformedURLException e )
		{
			System.err.println( "URL malformed." );
			e.printStackTrace( System.err );
			throw new RuntimeException(e);
		}
		catch ( final JsonSyntaxException e )
		{
			System.err.println( "JSON syntax malformed." );
			e.printStackTrace( System.err );
			throw new RuntimeException(e);
		}
		catch ( final Exception e )
		{
			e.printStackTrace( System.err );
			throw new RuntimeException(e);
		}
		return corr_data;
	}
	*/
	
	private static TileSpec[] readTileSpecFile( String tilespecFile )
	{

		final URL url;
		final TileSpec[] tileSpecs;
		
		try
		{
			final Gson gson = new Gson();
			url = new URL( tilespecFile );
			tileSpecs = gson.fromJson( new InputStreamReader( url.openStream() ), TileSpec[].class );
		}
		catch ( final MalformedURLException e )
		{
			System.err.println( "URL malformed." );
			e.printStackTrace( System.err );
			throw new RuntimeException(e);
		}
		catch ( final JsonSyntaxException e )
		{
			System.err.println( "JSON syntax malformed." );
			e.printStackTrace( System.err );
			throw new RuntimeException(e);
		}
		catch ( final Exception e )
		{
			e.printStackTrace( System.err );
			throw new RuntimeException(e);
		}
		
		return tileSpecs;
	}
	
	private static HashMap< Integer, List< TileSpec > > getTilesPerLayer(
			List< TileSpec > allTiles,
			int[] firstLastLayers )
	{
		final HashMap< Integer, List< TileSpec > > allLayers = new HashMap< Integer, List< TileSpec > >();

		firstLastLayers[0] = -1;
		firstLastLayers[1] = -1;
		
		for ( int i = 0; i < allTiles.size(); i++ )
		{
			TileSpec ts = allTiles.get( i );
			
			// Make sure that all tiles have a layer
			if ( ts.layer == -1 )
				throw new RuntimeException( "TileSpec " + i + " has no layer value" );
			
			if ( !allLayers.containsKey( ts.layer ) )
				allLayers.put( ts.layer, new ArrayList< TileSpec >() );
			
			List< TileSpec > layerList = allLayers.get( ts.layer );
			layerList.add( ts );
			
			if ( ( firstLastLayers[0] == -1 ) || ( ts.layer < firstLastLayers[0] ) )
				firstLastLayers[0] = ts.layer;
			if ( ( firstLastLayers[1] == -1 ) || ( ts.layer > firstLastLayers[1] ) )
				firstLastLayers[1] = ts.layer;
		}

		return allLayers;
	}
	
	private static ArrayList< Tile< ? > > createLayersModels( int layersNum, int desiredModelIndex )
	{
		/* create tiles and models for all layers */
		final ArrayList< Tile< ? > > tiles = new ArrayList< Tile< ? > >();
		for ( int i = 0; i < layersNum; ++i )
		{
			switch ( desiredModelIndex )
			{
			case 0:
				tiles.add( new Tile< TranslationModel2D >( new TranslationModel2D() ) );
				break;
			case 1:
				tiles.add( new Tile< RigidModel2D >( new RigidModel2D() ) );
				break;
			case 2:
				tiles.add( new Tile< SimilarityModel2D >( new SimilarityModel2D() ) );
				break;
			case 3:
				tiles.add( new Tile< AffineModel2D >( new AffineModel2D() ) );
				break;
			case 4:
				tiles.add( new Tile< HomographyModel2D >( new HomographyModel2D() ) );
				break;
			default:
				throw new RuntimeException( "Unknown desired model" );
			}
		}
		
		return tiles;
	}
	
	
	private static class ExtractFeaturesResult
	{
		public TileSpec ts;
		public List< Feature > features;
		
		public ExtractFeaturesResult( TileSpec ts, List< Feature > features )
		{
			this.ts = ts;
			this.features = features;
		}
	}
	
	/**
	 * Extract SIFT features and save them into the project folder.
	 * 
	 * @param layerSet the layerSet that contains all layers
	 * @param layerRange the list of layers to be aligned
	 * @param box a rectangular region of interest that will be used for alignment
	 * @param scale scale factor <= 1.0
	 * @param filter a name based filter for Patches (can be null)
	 * @param p SIFT extraction parameters
	 * @throws Exception
	 */
	final static protected HashMap< Integer, List< Feature > > extractLayersFeatures(
			final int mipmapLevel,
			final HashMap< Integer, List< TileSpec > > allLayers,
			final int startLayer,
			final int endLayer,
			final Params params,
			final int numThreads,
			final double scale ) throws ExecutionException, InterruptedException
	{
		final ExecutorService exec = Executors.newFixedThreadPool( numThreads );
		
		/* extract features for all slices and store them to disk */
		final AtomicInteger counter = new AtomicInteger( 0 );
		final List< Future< ExtractFeaturesResult > > siftTasks = new ArrayList< Future< ExtractFeaturesResult > >();
		
		//ComputeSiftFeatures.Params computeSiftParams = new ComputeSiftFeatures.Params();
		
		//final FloatArray2DSIFT.Param siftParam = ComputeSiftFeatures.generateSiftParams( computeSiftParams );
		final FloatArray2DSIFT.Param siftParam = new FloatArray2DSIFT.Param();
		siftParam.maxOctaveSize = params.sift_maxOctaveSize;
		siftParam.minOctaveSize = params.sift_minOctaveSize;
		siftParam.fdBins = params.sift_fdBins;
		siftParam.fdSize = params.sift_fdSize;
		siftParam.initialSigma = params.sift_initialSigma;
		siftParam.steps = params.sift_steps;

		HashMap< Integer, List< Feature > > sifts = new HashMap< Integer, List< Feature > >();
		
		for ( int i = startLayer; i <= endLayer; ++i )
		{
			final int layerIndex = i;
			final List< TileSpec > layerTileSpecs = allLayers.get( layerIndex );
			System.out.println( "Computing features for layer " + layerIndex + " with scale: " + scale );

//			for ( final TileSpec ts : layerTileSpecs )
//			{
//				siftTasks.add(
//					exec.submit( new Callable< ExtractFeaturesResult >()
//					{
//						@Override
//						public ExtractFeaturesResult call()
//						{
//							/* TODO: check if the features were already calculated */
//							final String imageUrl = ts.getMipmapLevels().get( String.valueOf( mipmapLevel ) ).imageUrl;
//							
//							final List< Feature > fs = ComputeSiftFeatures.computeTileSiftFeatures( imageUrl, siftParam );
//							final ExtractFeaturesResult res = new ExtractFeaturesResult( ts, fs );
//							return res;
//						}
//					} ) );
//			}
			
			TileSpecsImage layerImage = new TileSpecsImage( layerTileSpecs );
			ColorProcessor cp = layerImage.render( layerIndex, mipmapLevel, ( float )scale );
			final List< Feature > fs = ComputeSiftFeatures.computeImageSiftFeatures( cp, siftParam );
			System.out.println( "Found " + fs.size() + " features in the layer" );
			sifts.put( layerIndex, fs );
		}
		
		/* join */
//		try
//		{
//			for ( final Future< ExtractFeaturesResult > fu : siftTasks )
//			{
//				ExtractFeaturesResult result = fu.get();
//				int layer = result.ts.layer;
//				List< Feature > entry;
//				if ( sifts.containsKey( layer ) )
//				{
//					entry = sifts.get( layer );
//				}
//				else
//				{
//					entry = new ArrayList< Feature >();
//					sifts.put( layer, entry );
//				}
//				entry.addAll( result.features );
//			}
//		}
//		catch ( final InterruptedException e )
//		{
//			System.err.println( "Feature extraction interrupted." );
//			siftTasks.clear();
//			exec.shutdownNow();
//			throw e;
//		}
//		catch ( final ExecutionException e )
//		{
//			System.err.println( "Execution exception during feature extraction." );
//			siftTasks.clear();
//			exec.shutdownNow();
//			throw e;
//		}
//		
//		siftTasks.clear();
//		exec.shutdown();
		return sifts;
	}

	private static ArrayList< Triple< Integer, Integer, AbstractModel< ? > > > matchAndFilter(
			final Params param,
			final HashMap< Integer, List< TileSpec > > allLayers,
			final int endLayer,
			final double scale,
			final HashMap< Integer, List< Feature > > sifts
			)
	{
		/* collect all pairs of slices for which a model could be found */
		final ArrayList< Triple< Integer, Integer, AbstractModel< ? > > > pairs = new ArrayList< Triple< Integer, Integer, AbstractModel< ? > > >();

		/* match and filter feature correspondences */
		int numFailures = 0;
		
		final double pointMatchScale = param.layerScale / scale;
		
		
		for ( Integer layeri : allLayers.keySet() )
		{
			final ArrayList< Thread > threads = new ArrayList< Thread >( param.numThreads );
			
			final int sliceA = layeri.intValue();
			final List< TileSpec > layerA = allLayers.get( layeri );
			final int range = Math.min( endLayer + 1, layeri + param.maxNumNeighbors + 1 );
			
			final String layerNameA = "Section " + sliceA;
				
J:			for ( int j = layeri + 1; j < range; )
			{
				final int numThreads = Math.min( param.numThreads, range - j );
				final ArrayList< Triple< Integer, Integer, AbstractModel< ? > > > models =
					new ArrayList< Triple< Integer, Integer, AbstractModel< ? > > >( numThreads );
				
				for ( int k = 0; k < numThreads; ++k )
					models.add( null );
				
				for ( int t = 0;  t < numThreads && j < range; ++t, ++j )
				{
					final int ti = t;
					final int sliceB = j;
					final List< TileSpec > layerB = allLayers.get( j );
					
					final String layerNameB = "Section " + sliceB;
					
					final Thread thread = new Thread()
					{
						@Override
						public void run()
						{
							//IJ.showProgress( sliceA, layerRange.size() - 1 );
							
							System.out.println( "matching " + layerNameB + " -> " + layerNameA + "..." );
							
							ArrayList< PointMatch > candidates = null;
							/*
							if ( !param.ppm.clearCache )
								candidates = mpicbg.trakem2.align.Util.deserializePointMatches(
										project, param.ppm, "layer", layerB.getId(), layerA.getId() );
							*/
							
							if ( null == candidates )
							{
								final List< Feature > fs1 = sifts.get( sliceA );
								final List< Feature > fs2 = sifts.get( sliceB );
								/*
								final ArrayList< Feature > fs1 = mpicbg.trakem2.align.Util.deserializeFeatures(
										project, param.ppm.sift, "layer", layerA.getId() );
								final ArrayList< Feature > fs2 = mpicbg.trakem2.align.Util.deserializeFeatures(
										project, param.ppm.sift, "layer", layerB.getId() );
								*/
								candidates = new ArrayList< PointMatch >( FloatArray2DSIFT.createMatches( fs2, fs1, param.match_rod ) );
								
								/* scale the candidates */
								for ( final PointMatch pm : candidates )
								{
									final Point p1 = pm.getP1();
									final Point p2 = pm.getP2();
									final float[] l1 = p1.getL();
									final float[] w1 = p1.getW();
									final float[] l2 = p2.getL();
									final float[] w2 = p2.getW();
									
									l1[ 0 ] *= pointMatchScale;
									l1[ 1 ] *= pointMatchScale;
									w1[ 0 ] *= pointMatchScale;
									w1[ 1 ] *= pointMatchScale;
									l2[ 0 ] *= pointMatchScale;
									l2[ 1 ] *= pointMatchScale;
									w2[ 0 ] *= pointMatchScale;
									w2[ 1 ] *= pointMatchScale;
									
									System.out.println( "* Candidate: L(" + l1[0] + "," + l1[1] + ") -> L(" + l2[0] + "," + l2[1] + ")\t\t" +
											"W(" + w1[0] + "," + w1[1] + ") -> W(" + w2[0] + "," + w2[1] + ")");
								}
								
								/*
								if ( !mpicbg.trakem2.align.Util.serializePointMatches(
										project, param.ppm, "layer", layerB.getId(), layerA.getId(), candidates ) )
									Utils.log( "Could not store point match candidates for layers " + layerNameB + " and " + layerNameA + "." );
								*/
								System.out.println( "Found " + candidates.size() + " candidates when matching layers " + layerNameA + " and " + layerNameB );
							}
		
							AbstractModel< ? > model;
							switch ( param.ransac_modelIndex )
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
								return;
							}
							
							final ArrayList< PointMatch > inliers = new ArrayList< PointMatch >();
							
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
												param.maxEpsilon * param.layerScale,
												param.ransac_minInlierRatio,
												param.ransac_minNumInliers,
												3 );
									if ( modelFound && param.ransac_rejectIdentity )
									{
										final ArrayList< Point > points = new ArrayList< Point >();
										PointMatch.sourcePoints( inliers, points );
										if ( Transforms.isIdentity( model, points, param.ransac_identityTolerance *  param.layerScale ) )
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
								System.out.println( layerNameB + " -> " + layerNameA + ": " + inliers.size() + " corresponding features with an average displacement of " + ( PointMatch.meanDistance( inliers ) / param.layerScale ) + "px identified." );
								System.out.println( "Estimated transformation model: " + model );
								models.set( ti, new Triple< Integer, Integer, AbstractModel< ? > >( sliceA, sliceB, model ) );
							}
							else
							{
								System.out.println( layerNameB + " -> " + layerNameA + ": no correspondences found." );
								return;
							}
						}
					};
					threads.add( thread );
					thread.start();
				}
				
				try
				{
					for ( final Thread thread : threads )
						thread.join();
				}
				catch ( final InterruptedException e )
				{
					System.err.println( "Establishing feature correspondences interrupted." );
					//Utils.log( "Establishing feature correspondences interrupted." );
					for ( final Thread thread : threads )
						thread.interrupt();
					try
					{
						for ( final Thread thread : threads )
							thread.join();
					}
					catch ( final InterruptedException f ) {}
					throw new RuntimeException( e );
				}
				
				threads.clear();
				
				/* collect successfully matches pairs and break the search on gaps */
				for ( int t = 0; t < models.size(); ++t )
				{
					final Triple< Integer, Integer, AbstractModel< ? > > pair = models.get( t );
					if ( pair == null )
					{
						if ( ++numFailures > param.ransac_maxNumFailures )
							break J;
					}
					else
					{
						numFailures = 0;
						pairs.add( pair );
					}
				}
			}
		}

		return pairs;
	}
	
	/**
	 * Receives a single layer, applies the transformations, and outputs the ip and mask
	 * of the given level (render the ip and ipMask)
	 * 
	 * @param layerTileSpecs
	 * @param ip
	 * @param ipMask
	 * @param mipmapLevel
	 */
	private static void tilespecToFloatAndMask(
			final List< TileSpec > layerTileSpecs,
			final FloatProcessor output,
			final FloatProcessor alpha,
			final int mipmapLevel,
			final float layerScale )
	{
		final TileSpecsImage tsImg = new TileSpecsImage( layerTileSpecs );
		final int layer = layerTileSpecs.get( 0 ).layer;
		final ColorProcessor cp = tsImg.render( layer, mipmapLevel, layerScale );
		final int[] inputPixels = ( int[] )cp.getPixels();
		for ( int i = 0; i < inputPixels.length; ++i )
		{
			final int argb = inputPixels[ i ];
			final int a = ( argb >> 24 ) & 0xff;
			final int r = ( argb >> 16 ) & 0xff;
			final int g = ( argb >> 8 ) & 0xff;
			final int b = argb & 0xff;
			
			final float v = ( r + g + b ) / ( float )3;
			final float w = a / ( float )255;
			
			output.setf( i, v );
			alpha.setf( i, w );
		}
	}
	
	private static List< TileSpec > elasticLayerAlignment( 
			final Params param,
			final HashMap< Integer, List< TileSpec > > allLayers,
			final BoundingBox boundingBox,
			final int startLayer,
			final int endLayer,
			final int mipmapLevel,
			final List< Integer > fixedLayers )
	{
		final double scale = Math.min( 1.0, Math.min( ( double )param.sift_maxOctaveSize / ( double )boundingBox.getWidth(), ( double )param.sift_maxOctaveSize / ( double )boundingBox.getHeight() ) );

		final ArrayList< Tile< ? > > tiles = createLayersModels( endLayer - startLayer + 1, param.modelIndex );
		
		HashMap< Integer, List< Feature > > sifts;
		
		/* extract features */
		try
		{
			sifts = extractLayersFeatures( mipmapLevel, allLayers, startLayer, endLayer, param, param.numThreads, scale );
		}
		catch ( final Exception e)
		{
			System.err.println( "Sift computation interrupted." );
			throw new RuntimeException( e );
		}
/*		try
		{
			AlignmentUtils.extractAndSaveLayerFeatures( allLayers, box, scale, filter, param.ppm.sift, param.ppm.clearCache, param.ppm.maxNumThreadsSift );
		}
		catch ( final Exception e )
		{
			return;
		}
*/

		/* collect all pairs of slices for which a model could be found */
		final ArrayList< Triple< Integer, Integer, AbstractModel< ? > > > pairs = matchAndFilter(
				param,
				allLayers,
				endLayer,
				scale,
				sifts);
		

		/* Free memory */
		sifts = null;
		System.gc();
		System.gc();

		/* Elastic alignment */
		
		/* Initialization */
		final TileConfiguration initMeshes = new TileConfiguration();
		
		final int meshWidth = ( int )Math.ceil( boundingBox.getWidth() * param.layerScale );
		final int meshHeight = ( int )Math.ceil( boundingBox.getHeight() * param.layerScale );
		
		final ArrayList< SpringMesh > meshes = new ArrayList< SpringMesh >( endLayer - startLayer + 1 );
		for ( int i = startLayer; i <= endLayer; ++i )
			meshes.add(
					new SpringMesh(
							param.resolutionSpringMesh,
							meshWidth,
							meshHeight,
							param.stiffnessSpringMesh,
							param.maxStretchSpringMesh * param.layerScale,
							param.dampSpringMesh ) );
		
		//final int blockRadius = Math.max( 32, meshWidth / p.resolutionSpringMesh / 2 );
		final int param_blockRadius = boundingBox.getWidth() / param.pmcc_resolution / 2;
		final int blockRadius = Math.max( 16, mpicbg.util.Util.roundPos( param.layerScale * param_blockRadius ) );
		
		System.out.println( "effective block radius = " + blockRadius );
		
		/* scale pixel distances */
		final int searchRadius = ( int )Math.round( param.layerScale * param.pmcc_searchRadius );
		final float localRegionSigma = param.layerScale * param.pmcc_localRegionSigma;
		final float maxLocalEpsilon = param.layerScale * param.pmcc_maxLocalEpsilon;
		
		final AbstractModel< ? > localSmoothnessFilterModel = Util.createModel( param.pmcc_localModelIndex );
		
		
		for ( final Triple< Integer, Integer, AbstractModel< ? > > pair : pairs )
		{
			/* free memory */
			/*
			project.getLoader().releaseAll();
			*/
			
			final SpringMesh m1 = meshes.get( pair.a - startLayer );
			final SpringMesh m2 = meshes.get( pair.b - startLayer );

			final ArrayList< PointMatch > pm12 = new ArrayList< PointMatch >();
			final ArrayList< PointMatch > pm21 = new ArrayList< PointMatch >();

			final ArrayList< Vertex > v1 = m1.getVertices();
			final ArrayList< Vertex > v2 = m2.getVertices();
			
			final List< TileSpec > layer1 = allLayers.get( pair.a );
			final List< TileSpec > layer2 = allLayers.get( pair.b );
			
			final boolean layer1Fixed = fixedLayers.contains( pair.a );
			final boolean layer2Fixed = fixedLayers.contains( pair.b );
			
			final Tile< ? > t1 = tiles.get( pair.a - startLayer );
			final Tile< ? > t2 = tiles.get( pair.b - startLayer );
			
			if ( !( layer1Fixed && layer2Fixed ) )
			{
				/* Load images and masks into FloatProcessor objects */
				/*
				final Image img1 = project.getLoader().getFlatAWTImage(
						layer1,
						box,
						param.layerScale,
						0xffffffff,
						ImagePlus.COLOR_RGB,
						Patch.class,
						AlignmentUtils.filterPatches( layer1, filter ),
						true,
						new Color( 0x00ffffff, true ) );
				
				final Image img2 = project.getLoader().getFlatAWTImage(
						layer2,
						box,
						param.layerScale,
						0xffffffff,
						ImagePlus.COLOR_RGB,
						Patch.class,
						AlignmentUtils.filterPatches( layer2, filter ),
						true,
						new Color( 0x00ffffff, true ) );
				*/

				/*
				final Image img1;
				final Image img2;

				img1 = Utils.openImageUrl(allLayers.get( pair.a ).get( 0 ).getMipmapLevels().get( "0" ).imageUrl);
				img2 = Utils.openImageUrl(allLayers.get( pair.b ).get( 0 ).getMipmapLevels().get( "0" ).imageUrl);
				
				final int width = img1.getWidth( null );
				final int height = img1.getHeight( null );
	
				final FloatProcessor ip1 = new FloatProcessor( width, height );
				final FloatProcessor ip2 = new FloatProcessor( width, height );
				final FloatProcessor ip1Mask = new FloatProcessor( width, height );
				final FloatProcessor ip2Mask = new FloatProcessor( width, height );
				
				mpicbg.trakem2.align.Util.imageToFloatAndMask( img1, ip1, ip1Mask );
				mpicbg.trakem2.align.Util.imageToFloatAndMask( img2, ip2, ip2Mask );
				*/

				
				final TileSpecsImage layer1Img = new TileSpecsImage( layer1 );
				final TileSpecsImage layer2Img = new TileSpecsImage( layer2 );
				
				final BoundingBox layer1BBox = layer1Img.getBoundingBox();
				final BoundingBox layer2BBox = layer2Img.getBoundingBox();
				
				final int img1Width = (int) (layer1BBox.getWidth() * param.layerScale);
				final int img1Height = (int) (layer1BBox.getHeight() * param.layerScale);
				final int img2Width = (int) (layer2BBox.getWidth() * param.layerScale);
				final int img2Height = (int) (layer2BBox.getHeight() * param.layerScale);

				final FloatProcessor ip1 = new FloatProcessor( img1Width, img1Height );
				final FloatProcessor ip2 = new FloatProcessor( img2Width, img2Height );
				final FloatProcessor ip1Mask = new FloatProcessor( img1Width, img1Height );
				final FloatProcessor ip2Mask = new FloatProcessor( img2Width, img2Height );

				// TODO: load the tile specs to FloatProcessor objects
				tilespecToFloatAndMask( layer1, ip1, ip1Mask, mipmapLevel, param.layerScale );
				tilespecToFloatAndMask( layer2, ip2, ip2Mask, mipmapLevel, param.layerScale );
				
				final float springConstant  = 1.0f / ( pair.b - pair.a );
				
				if ( layer1Fixed )
					initMeshes.fixTile( t1 );
				else
				{
					try
					{
						BlockMatching.matchByMaximalPMCC(
								ip1,
								ip2,
								null,//ip1Mask,
								null,//ip2Mask,
								1.0f,
								( ( InvertibleCoordinateTransform )pair.c ).createInverse(),
								blockRadius,
								blockRadius,
								searchRadius,
								searchRadius,
								param.pmcc_minR,
								param.pmcc_rodR,
								param.pmcc_maxCurvatureR,
								v1,
								pm12,
								new ErrorStatistic( 1 ) );
					}
					catch ( final InterruptedException e )
					{
						System.err.println( "Block matching interrupted." );
						//IJ.showProgress( 1.0 );
						throw new RuntimeException( e );
					}
					catch ( final ExecutionException e )
					{
						System.out.println( "Block matching interrupted." );
						throw new RuntimeException( e );
					}
					if ( Thread.interrupted() )
					{
						System.err.println( "Block matching interrupted." );
						//IJ.showProgress( 1.0 );
						throw new RuntimeException( "Block matching interrupted." );
					}
		
					if ( param.pmcc_useLocalSmoothnessFilter )
					{
						System.out.println( pair.a + " > " + pair.b + ": found " + pm12.size() + " correspondence candidates." );
						localSmoothnessFilterModel.localSmoothnessFilter( pm12, pm12, localRegionSigma, maxLocalEpsilon, param.pmcc_maxLocalTrust );
						System.out.println( pair.a + " > " + pair.b + ": " + pm12.size() + " candidates passed local smoothness filter." );
					}
					else
					{
						System.out.println( pair.a + " > " + pair.b + ": found " + pm12.size() + " correspondences." );
					}
		
					/* <visualisation> */
					//			final List< Point > s1 = new ArrayList< Point >();
					//			PointMatch.sourcePoints( pm12, s1 );
					//			final ImagePlus imp1 = new ImagePlus( i + " >", ip1 );
					//			imp1.show();
					//			imp1.setOverlay( BlockMatching.illustrateMatches( pm12 ), Color.yellow, null );
					//			imp1.setRoi( Util.pointsToPointRoi( s1 ) );
					//			imp1.updateAndDraw();
					/* </visualisation> */
					
					for ( final PointMatch pm : pm12 )
					{
						final Vertex p1 = ( Vertex )pm.getP1();
						final Vertex p2 = new Vertex( pm.getP2() );
						p1.addSpring( p2, new Spring( 0, springConstant ) );
						m2.addPassiveVertex( p2 );
					}
					
					/*
					 * adding Tiles to the initialing TileConfiguration, adding a Tile
					 * multiple times does not harm because the TileConfiguration is
					 * backed by a Set. 
					 */
					if ( pm12.size() > pair.c.getMinNumMatches() )
					{
						initMeshes.addTile( t1 );
						initMeshes.addTile( t2 );
						t1.connect( t2, pm12 );
					}
				}
	
				if ( layer2Fixed )
					initMeshes.fixTile( t2 );
				else
				{
					try
					{
						BlockMatching.matchByMaximalPMCC(
								ip2,
								ip1,
								null,//ip2Mask,
								null,//ip1Mask,
								1.0f,
								pair.c,
								blockRadius,
								blockRadius,
								searchRadius,
								searchRadius,
								param.pmcc_minR,
								param.pmcc_rodR,
								param.pmcc_maxCurvatureR,
								v2,
								pm21,
								new ErrorStatistic( 1 ) );
					}
					catch ( final InterruptedException e )
					{
						System.out.println( "Block matching interrupted." );
						//IJ.showProgress( 1.0 );
						throw new RuntimeException( e );
					}
					catch ( final ExecutionException e )
					{
						System.out.println( "Block matching interrupted." );
						throw new RuntimeException( e );
					}
					if ( Thread.interrupted() )
					{
						System.out.println( "Block matching interrupted." );
						//IJ.showProgress( 1.0 );
						throw new RuntimeException( "Block matching interrupted." );
					}
		
					if ( param.pmcc_useLocalSmoothnessFilter )
					{
						System.out.println( pair.a + " < " + pair.b + ": found " + pm21.size() + " correspondence candidates." );
						localSmoothnessFilterModel.localSmoothnessFilter( pm21, pm21, localRegionSigma, maxLocalEpsilon, param.pmcc_maxLocalTrust );
						System.out.println( pair.a + " < " + pair.b + ": " + pm21.size() + " candidates passed local smoothness filter." );
					}
					else
					{
						System.out.println( pair.a + " < " + pair.b + ": found " + pm21.size() + " correspondences." );
					}
					
					/* <visualisation> */
					//			final List< Point > s2 = new ArrayList< Point >();
					//			PointMatch.sourcePoints( pm21, s2 );
					//			final ImagePlus imp2 = new ImagePlus( i + " <", ip2 );
					//			imp2.show();
					//			imp2.setOverlay( BlockMatching.illustrateMatches( pm21 ), Color.yellow, null );
					//			imp2.setRoi( Util.pointsToPointRoi( s2 ) );
					//			imp2.updateAndDraw();
					/* </visualisation> */
					
					for ( final PointMatch pm : pm21 )
					{
						final Vertex p1 = ( Vertex )pm.getP1();
						final Vertex p2 = new Vertex( pm.getP2() );
						p1.addSpring( p2, new Spring( 0, springConstant ) );
						m1.addPassiveVertex( p2 );
					}
					
					/*
					 * adding Tiles to the initialing TileConfiguration, adding a Tile
					 * multiple times does not harm because the TileConfiguration is
					 * backed by a Set. 
					 */
					if ( pm21.size() > pair.c.getMinNumMatches() )
					{
						initMeshes.addTile( t1 );
						initMeshes.addTile( t2 );
						t2.connect( t1, pm21 );
					}
				}
				
				System.out.println( pair.a + " <> " + pair.b + " spring constant = " + springConstant );
			}
		}
		
		/* pre-align by optimizing a piecewise linear model */
		try
		{
			initMeshes.optimize(
					param.maxEpsilon * param.layerScale,
					param.maxIterationsSpringMesh,
					param.maxPlateauwidthSpringMesh );
		}
		catch ( Exception e )
		{
			throw new RuntimeException( e );
		}
		for ( int i = startLayer; i <= endLayer; ++i )
			meshes.get( i - startLayer ).init( tiles.get( i - startLayer ).getModel() );
		
		/* optimize the meshes */
		try
		{
			final long t0 = System.currentTimeMillis();
			System.out.println( "Optimizing spring meshes..." );
			
			if ( param.useLegacyOptimizer )
			{
				System.out.println( "  ...using legacy optimizer...");
				SpringMesh.optimizeMeshes2(
						meshes,
						param.maxEpsilon * param.layerScale,
						param.maxIterationsSpringMesh,
						param.maxPlateauwidthSpringMesh );
			}
			else
			{
				SpringMesh.optimizeMeshes(
						meshes,
						param.maxEpsilon * param.layerScale,
						param.maxIterationsSpringMesh,
						param.maxPlateauwidthSpringMesh );
			}

			System.out.println( "Done optimizing spring meshes. Took " + (System.currentTimeMillis() - t0) + " ms");
			
		}
		catch ( final NotEnoughDataPointsException e )
		{
			System.err.println( "There were not enough data points to get the spring mesh optimizing." );
			e.printStackTrace();
			throw new RuntimeException( e );
		}
		
		/* translate relative to bounding box */
		final int boxX = boundingBox.getStartPoint().getX();
		final int boxY = boundingBox.getStartPoint().getY();
		for ( final SpringMesh mesh : meshes )
		{
			for ( final PointMatch pm : mesh.getVA().keySet() )
			{
				final Point p1 = pm.getP1();
				final Point p2 = pm.getP2();
				final float[] l = p1.getL();
				final float[] w = p2.getW();
				l[ 0 ] = l[ 0 ] / param.layerScale + boxX;
				l[ 1 ] = l[ 1 ] / param.layerScale + boxY;
				w[ 0 ] = w[ 0 ] / param.layerScale + boxX;
				w[ 1 ] = w[ 1 ] / param.layerScale + boxY;
			}
		}
		
		
		/* free memory */
		/*
		project.getLoader().releaseAll();
		*/
		
//		final Layer first = layerRange.get( 0 );
//		final List< Layer > layers = first.getParent().getLayers();
//
//		/* transfer layer transform into patch transforms and append to patches */
//		if ( propagateTransformBefore || propagateTransformAfter )
//		{
//			if ( propagateTransformBefore )
//			{
//				final MovingLeastSquaresTransform2 mlt = makeMLST2( meshes.get( 0 ).getVA().keySet() );
//				final int firstLayerIndex = first.getParent().getLayerIndex( first.getId() );
//				for ( int i = 0; i < firstLayerIndex; ++i )
//					applyTransformToLayer( layers.get( i ), mlt, filter );
//			}
//			if ( propagateTransformAfter )
//			{
//				final Layer last = layerRange.get( layerRange.size() - 1 );
//				final MovingLeastSquaresTransform2 mlt = makeMLST2( meshes.get( meshes.size() - 1 ).getVA().keySet() );
//				final int lastLayerIndex = last.getParent().getLayerIndex( last.getId() );
//				for ( int i = lastLayerIndex + 1; i < layers.size(); ++i )
//					applyTransformToLayer( layers.get( i ), mlt, filter );
//			}
//		}
//		for ( int l = 0; l < layerRange.size(); ++l )
//		{
//			IJ.showStatus( "Applying transformation to patches ..." );
//			IJ.showProgress( 0, layerRange.size() );
//			
//			final Layer layer = layerRange.get( l );
//			
//			final MovingLeastSquaresTransform2 mlt = new MovingLeastSquaresTransform2();
//			mlt.setModel( AffineModel2D.class );
//			mlt.setAlpha( 2.0f );
//			mlt.setMatches( meshes.get( l ).getVA().keySet() );
//			
//			applyTransformToLayer( layer, mlt, filter );
//					
//			if ( Thread.interrupted() )
//			{
//				Utils.log( "Interrupted during applying transformations to patches.  No all patches have been updated.  Re-generate mipmaps manually." );
//			}
//			
//			IJ.showProgress( l + 1, layerRange.size() );
//		}
//		
//		/* update patch mipmaps */
//		final int firstLayerIndex;
//		final int lastLayerIndex;
//		
//		if ( propagateTransformBefore )
//			firstLayerIndex = 0;
//		else
//		{
//			firstLayerIndex = first.getParent().getLayerIndex( first.getId() );
//		}
//		if ( propagateTransformAfter )
//			 lastLayerIndex = layers.size() - 1;
//		else
//		{
//			final Layer last = layerRange.get( layerRange.size() - 1 );
//			lastLayerIndex = last.getParent().getLayerIndex( last.getId() );
//		}
//		
//		for ( int i = firstLayerIndex; i <= lastLayerIndex; ++i )
//		{
//			final Layer layer = layers.get( i );
//			if ( !( emptyLayers.contains( layer ) || fixedLayers.contains( layer ) ) )
//			{
//				for ( final Patch patch : AlignmentUtils.filterPatches( layer, filter ) )
//					patch.updateMipMaps();
//			}
//		}
//		
//		Utils.log( "Done." );
		
		// TODO: Save the optimized transformation into the tilespecs
		System.out.println( "Optimization complete. Generating tile transforms.");
		
		List< TileSpec > outTiles = new ArrayList< TileSpec >();
		
		/* calculate bounding box */
		/*
		final float[] min = new float[ 2 ];
		final float[] max = new float[ 2 ];
		for ( final SpringMesh mesh : meshes )
		{
			final float[] meshMin = new float[ 2 ];
			final float[] meshMax = new float[ 2 ];

			mesh.bounds( meshMin, meshMax );

			Utils.min( min, meshMin );
			Utils.max( max, meshMax );
		}
		*/
		/* translate relative to bounding box */
		/*
		for ( final SpringMesh mesh : meshes )
		{
			for ( final Vertex vertex : mesh.getVertices() )
			{
				final float[] w = vertex.getW();
				w[ 0 ] -= min[ 0 ];
				w[ 1 ] -= min[ 1 ];
			}
			mesh.updateAffines();
			mesh.updatePassiveVertices();
		}

		final int fullWidth = ( int )Math.ceil( max[ 0 ] - min[ 0 ] );
		final int fullHeight = ( int )Math.ceil( max[ 1 ] - min[ 1 ] );
		*/
		
		/* translate coordinate system to 0,0 */
		final float[] minXY = { Float.MAX_VALUE, Float.MAX_VALUE };
		for ( final SpringMesh mesh : meshes )
		{
			final float[] meshMin = new float[ 2 ];
			final float[] meshMax = new float[ 2 ];

			mesh.bounds( meshMin, meshMax );
			
			Utils.min( minXY, meshMin );
		}

		if ( ( minXY[0] != 0 ) || ( minXY[1] != 0 ) )
		{
			for ( final SpringMesh mesh : meshes )
			{
	
				for ( final PointMatch pm : mesh.getVA().keySet() )
				{
					final Point p1 = pm.getP1();
					final Point p2 = pm.getP2();
					final float[] l = p1.getL();
					final float[] w = p2.getW();
					l[ 0 ] = l[ 0 ] - minXY[0];
					l[ 1 ] = l[ 1 ] - minXY[1];
					w[ 0 ] = w[ 0 ] - minXY[0];
					w[ 1 ] = w[ 1 ] - minXY[1];
				}
			}
		}
		
		// Iterate the layers, and add the mesh transform for each tile
		// TODO: this might need to be done per tile-spec and not per layer
		for ( int i = startLayer; i <= endLayer; ++i )
		{
			final SpringMesh mesh = meshes.get( i - startLayer );
			final List< TileSpec > layer = allLayers.get( i );
			
			System.out.println( "Exporting tiles in layer " + i );
			
			for ( TileSpec ts : layer )
			{
				final String tileUrl = ts.getMipmapLevels().get( String.valueOf( mipmapLevel ) ).imageUrl;
				// bounding box after transformations are applied [left, right, top, bottom] possibly with extra entries for [front, back, etc.]
				final float[] meshMin = new float[ 2 ];
				final float[] meshMax = new float[ 2 ];
				mesh.bounds( meshMin, meshMax );			
				ts.bbox = new float[] {
						/*meshMin[0] * param.layerScale,
						meshMax[0] * param.layerScale,
						meshMin[1] * param.layerScale,
						meshMax[1] * param.layerScale */
						meshMin[0],
						meshMax[0],
						meshMin[1],
						meshMax[1]
					};
				

				try
				{
					final MovingLeastSquaresTransform mlt = new MovingLeastSquaresTransform();
					mlt.setModel( AffineModel2D.class );
					mlt.setAlpha( 2.0f );
					mlt.setMatches( mesh.getVA().keySet() );
		
				    Transform addedTransform = new Transform();				    
				    addedTransform.className = mlt.getClass().getCanonicalName().toString();
				    addedTransform.dataString = mlt.toDataString();
				    
					ArrayList< Transform > outTransforms = new ArrayList< Transform >(Arrays.asList(ts.transforms));
					outTransforms.add(addedTransform);
					ts.transforms = outTransforms.toArray(ts.transforms);
				}
				catch ( final Exception e )
				{
					System.out.println( "Error applying transform to tile " + tileUrl + "." );
					e.printStackTrace();
				}

			    outTiles.add(ts);
				
			}
		}
		
		return outTiles;
	}

	
	public static void main( String[] args )
	{
		final Params params = parseParams( args );

		/*
		final CorrespondenceSpec[] corr_data = readCorrespondenceFile( params.inputfile );
		*/
		
		// The mipmap level to work on
		// TODO: Should be a parameter from the user,
		//       and decide whether or not to create the mipmaps if they are missing
		int mipmapLevel = 0;


		/* read all tilespecs */
//		final HashMap< String, TileSpec > tileSpecMap = new HashMap< String, TileSpec >();

		List< TileSpec > tileSpecs = new ArrayList< TileSpec >();
		
		for ( String fileName : params.files )
		{
			TileSpec[] fileTileSpecs = readTileSpecFile( fileName );
			List< TileSpec > fileTileSpecsList = Arrays.asList( fileTileSpecs );
			tileSpecs.addAll( fileTileSpecsList );
		}
		

//		for (TileSpec ts : tileSpecs)
//		{
//			String imageUrl = ts.getMipmapLevels().get( String.valueOf( mipmapLevel ) ).imageUrl;
//			tileSpecMap.put(imageUrl, ts);
//		}

		final int[] firstLastLayers = new int[2];
		final HashMap< Integer, List< TileSpec > > allLayers = getTilesPerLayer( tileSpecs, firstLastLayers );
		
		
		final TileSpecsImage tsImage = new TileSpecsImage( tileSpecs );
		BoundingBox bbox = tsImage.getBoundingBox();
		
		final List< Integer > fixedLayers = new ArrayList< Integer >();
		fixedLayers.add( firstLastLayers[0] );
		List< TileSpec > outTiles = elasticLayerAlignment( params, allLayers,
				bbox, firstLastLayers[0], firstLastLayers[1], mipmapLevel, fixedLayers );
		
		System.out.println( "Exporting tiles to output json file.");
		
		// Consolidate each layer tile specs
		final HashMap< Integer, List< TileSpec > > allOutLayers = new HashMap<Integer, List<TileSpec>>();
		for ( final TileSpec ts : outTiles )
		{
			final int layer = ts.layer;
			final List< TileSpec > tsList;
			if ( allOutLayers.containsKey( layer ) )
			{
				tsList = allOutLayers.get( layer );
			}
			else
			{
				tsList = new ArrayList<TileSpec>();
				allOutLayers.put( layer, tsList );
			}
			tsList.add( ts );
		}
		
		for ( Integer layer : allOutLayers.keySet() )
		{
			String layerString = String.format( "%03d", layer );
			System.out.println( "Writing layer " + layerString );
			final File targetFile = new File( params.targetDir, "Section" + layerString + ".json" );
			final List< TileSpec > layerOutTiles = allOutLayers.get( layer );
			try {
				Writer writer = new FileWriter(targetFile);
		        //Gson gson = new GsonBuilder().create();
		        Gson gson = new GsonBuilder().setPrettyPrinting().create();
		        gson.toJson(layerOutTiles, writer);
		        writer.close();
		    }
			catch ( final IOException e )
			{
				System.err.println( "Error writing JSON file: " + targetFile.getAbsolutePath() );
				e.printStackTrace( System.err );
			}
		}
		System.out.println( "Done." );

	}
	
}

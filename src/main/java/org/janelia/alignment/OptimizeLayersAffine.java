package org.janelia.alignment;

import java.io.BufferedReader;
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
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import mpicbg.models.AbstractAffineModel2D;
import mpicbg.models.AbstractModel;
import mpicbg.models.Affine2D;
import mpicbg.models.AffineModel2D;
import mpicbg.models.HomographyModel2D;
import mpicbg.models.Model;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.RigidModel2D;
import mpicbg.models.SimilarityModel2D;
import mpicbg.models.Spring;
import mpicbg.models.SpringMesh;
import mpicbg.models.SpringMeshConcurrent;
import mpicbg.models.Tile;
import mpicbg.models.TranslationModel2D;
import mpicbg.models.Vertex;
import mpicbg.trakem2.transform.CoordinateTransform;
import mpicbg.trakem2.transform.CoordinateTransformList;
import mpicbg.trakem2.transform.MovingLeastSquaresTransform2;
import mpicbg.trakem2.transform.RestrictedMovingLeastSquaresTransform2;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;

public class OptimizeLayersAffine {
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--corrFiles", description = "Correspondence json files  (space separated) or a single file containing a line-separated list of json files", variableArity = true, required = true )
        public List<String> corrFiles = new ArrayList<String>();

        @Parameter( names = "--modelFiles", description = "Model json files  (space separated) or a single file containing a line-separated list of json files", variableArity = true, required = true )
        public List<String> modelFiles = new ArrayList<String>();

        @Parameter( names = "--tilespecFiles", description = "Tilespec json files  (space separated) or a single file containing a line-separated list of json files", variableArity = true, required = true )
        public List<String> tileSpecFiles = new ArrayList<String>();
        
        @Parameter( names = "--fixedLayers", description = "Fixed layer numbers (space separated)", variableArity = true, required = true )
        public List<Integer> fixedLayers = new ArrayList<Integer>();
        
        @Parameter( names = "--targetDir", description = "Directory to output the new tilespec files", required = true )
        public String targetDir;

        @Parameter( names = "--maxLayersDistance", description = "The number of neighboring layers to match", required = false )
        private int maxLayersDistance;               

        @Parameter( names = "--modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int modelIndex = 1;
        
        @Parameter( names = "--maxEpsilon", description = "maxEpsilon", required = false )
        private float maxEpsilon = 200.0f;
        
        @Parameter( names = "--maxIterations", description = "maxIterations", required = false )
        private int maxIterations = 1000;
        
        @Parameter( names = "--maxPlateauwidth", description = "maxPlateauwidth", required = false )
        private int maxPlateauwidth = 200;
        
        @Parameter( names = "--ransacModelIndex", description = "Model Index that was used for ransac phase: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int ransacModelIndex = 3;

        @Parameter( names = "--ransacMaxEpsilon", description = "Max epsilon that was used for ransac phase", required = false )
        private float ransacMaxEpsilon = 200.0f;

        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
        @Parameter( names = "--fromLayer", description = "The layer to start the optimization from (default: first layer in the tile specs data)", required = false )
        private int fromLayer = -1;

        @Parameter( names = "--toLayer", description = "The last layer to include in the optimization (default: last layer in the tile specs data)", required = false )
        private int toLayer = -1;
               
        @Parameter( names = "--skipLayers", description = "The layers ranges that will not be processed (default: none)", required = false )
        private String skippedLayers = "";

        @Parameter( names = "--manualMatches", description = "Pair of layer indices (each pair is separated by a colon) that need to be manually aligned with a spring of constant of 1", required = false )
        public List<String> manualMatches = new ArrayList<String>();
	}
	
	private OptimizeLayersAffine() {}
	
	private static Map< Integer, Map< Integer, CorrespondenceSpec > > parseCorrespondenceFiles(
			final List< String > fileUrls,
			final HashMap< String, Integer > tsUrlToLayerIds )
	{
		System.out.println( "Parsing correspondence files" );
		Map< Integer, Map< Integer, CorrespondenceSpec > > layersCorrs = new HashMap<Integer, Map<Integer,CorrespondenceSpec>>();
		
		for ( String fileUrl : fileUrls )
		{
			try
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

				if ( corr_data.length == 0 ) // No correspondences, skip this file
					continue;
				
				assert( corr_data.length == 1 );
				
				CorrespondenceSpec corr = corr_data[ 0 ];
				final int layer1Id;
				final int layer2Id;
				if ( tsUrlToLayerIds.containsKey( corr.url1 ) )
					layer1Id = tsUrlToLayerIds.get( corr.url1 );
				else
					layer1Id = readLayerFromFile( corr.url1 );
				if ( tsUrlToLayerIds.containsKey( corr.url2 ) )
					layer2Id = tsUrlToLayerIds.get( corr.url2 );
				else
					layer2Id = readLayerFromFile( corr.url2 );

				final Map< Integer, CorrespondenceSpec > innerMapping;

				int minLayerId = Math.min( layer1Id, layer2Id );
				int maxLayerId = Math.max( layer1Id, layer2Id );
				
				if ( layersCorrs.containsKey( minLayerId ) )
				{
					innerMapping = layersCorrs.get( minLayerId );
				}
				else
				{
					innerMapping = new HashMap<Integer, CorrespondenceSpec>();
					layersCorrs.put( minLayerId, innerMapping );
				}
				// Assuming that no two files have the same correspondence spec url values
				innerMapping.put( maxLayerId,  corr );
			}
			catch (RuntimeException e)
			{
				System.err.println( "Error while reading file: " + fileUrl );
				e.printStackTrace( System.err );
				throw e;
			}
		}
		
		return layersCorrs;
	}

	private static int readLayerFromFile( String tsUrl )
	{
		final TileSpec[] tileSpecs = TileSpecUtils.readTileSpecFile( tsUrl );
		int layer = tileSpecs[0].layer;
		if ( layer == -1 )
			throw new RuntimeException( "Error: a tile spec json file (" + tsUrl + ") has a tilespec without a layer " );
		return layer;
	}
	

	private static Map< Integer, Map< Integer, CorrespondenceSpec > > parseCorrespondenceFiles(
			final List< String > fileUrls,
			final HashMap< String, Integer > tsUrlToLayerIds,
			final int threadsNum )
	{
		System.out.println( "Parsing correspondence files with " + threadsNum + " threads" );
		
		// Single thread case
		if ( threadsNum == 1 )
		{
			return parseCorrespondenceFiles( fileUrls, tsUrlToLayerIds );
		}
		
		final ConcurrentHashMap< Integer, Map< Integer, CorrespondenceSpec > > layersCorrs = new ConcurrentHashMap<Integer, Map<Integer,CorrespondenceSpec>>();
		
		// Initialize threads
		final ExecutorService exec = Executors.newFixedThreadPool( threadsNum );
		final ArrayList< Future< ? > > tasks = new ArrayList< Future< ? > >();

		final int filesPerThreadNum = fileUrls.size() / threadsNum;
		for ( int i = 0; i < threadsNum; i++ )
		{
			final int fromIndex = i * filesPerThreadNum;
			final int lastIndex;
			if ( i == threadsNum - 1 ) // lastThread
				lastIndex = fileUrls.size();
			else
				lastIndex = fromIndex + filesPerThreadNum;

			tasks.add( exec.submit( new Runnable() {
				
				@Override
				public void run() {
					// TODO Auto-generated method stub
					
					for ( int i = fromIndex; i < lastIndex; i++ )
					{
						final String fileUrl = fileUrls.get( i );
						try
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
							
							if ( corr_data.length == 0 ) // No correspondences, skip this file
								continue;
							
							assert( corr_data.length == 1 );
							
							CorrespondenceSpec corr = corr_data[ 0 ];
							final int layer1Id;
							final int layer2Id;
							if ( tsUrlToLayerIds.containsKey( corr.url1 ) )
								layer1Id = tsUrlToLayerIds.get( corr.url1 );
							else
								layer1Id = readLayerFromFile( corr.url1 );
							if ( tsUrlToLayerIds.containsKey( corr.url2 ) )
								layer2Id = tsUrlToLayerIds.get( corr.url2 );
							else
								layer2Id = readLayerFromFile( corr.url2 );

							int minLayerId = Math.min( layer1Id, layer2Id );
							int maxLayerId = Math.max( layer1Id, layer2Id );

							Map< Integer, CorrespondenceSpec > innerMapping;
							
							if ( layersCorrs.containsKey( minLayerId ) )
							{
								innerMapping = layersCorrs.get( minLayerId );
							}
							else
							{
								innerMapping = new ConcurrentHashMap<Integer, CorrespondenceSpec>();
								Map<Integer, CorrespondenceSpec> curValue = layersCorrs.putIfAbsent( minLayerId, innerMapping );
								// If by the time we executed put, some other thread already put something instead 
								if ( curValue != null )
									innerMapping = layersCorrs.get( minLayerId );
							}
							// Assuming that no two files have the same correspondence spec url values
							innerMapping.put( maxLayerId,  corr );
						}
						catch (RuntimeException e)
						{
							System.err.println( "Error while reading file: " + fileUrl );
							e.printStackTrace( System.err );
							throw e;
						}
					}

				}
			}));

			
		}

		for ( Future< ? > task : tasks )
		{
			try {
				task.get();
			} catch (InterruptedException e) {
				exec.shutdownNow();
				e.printStackTrace();
			} catch (ExecutionException e) {
				exec.shutdownNow();
				e.printStackTrace();
			}
		}
		exec.shutdown();

		
		return layersCorrs;
	}

	private static Map< Integer, Map< Integer, ModelSpec > > parseModelFiles(
			final List< String > fileUrls,
			final HashMap< String, Integer > tsUrlToLayerIds )
	{
		System.out.println( "Parsing model files" );
		Map< Integer, Map< Integer, ModelSpec > > layersModels = new HashMap<Integer, Map<Integer,ModelSpec>>();
		
		for ( String fileUrl : fileUrls )
		{
			try
			{
				// Open and parse the json file
				final ModelSpec[] model_data;
				try
				{
					final Gson gson = new Gson();
					URL url = new URL( fileUrl );
					model_data = gson.fromJson( new InputStreamReader( url.openStream() ), ModelSpec[].class );
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
	
				if ( model_data.length == 0 ) // No correspondences, skip this file
					continue;
				
				assert( model_data.length == 1 );
				
				ModelSpec mspec = model_data[ 0 ];
				final int layer1Id;
				final int layer2Id;
				if ( tsUrlToLayerIds.containsKey( mspec.url1 ) )
					layer1Id = tsUrlToLayerIds.get( mspec.url1 );
				else
					layer1Id = readLayerFromFile( mspec.url1 );
				if ( tsUrlToLayerIds.containsKey( mspec.url2 ) )
					layer2Id = tsUrlToLayerIds.get( mspec.url2 );
				else
					layer2Id = readLayerFromFile( mspec.url2 );

				int minLayerId = Math.min( layer1Id, layer2Id );
				int maxLayerId = Math.max( layer1Id, layer2Id );

				final Map< Integer, ModelSpec > innerMapping;

				if ( layersModels.containsKey( minLayerId ) )
				{
					innerMapping = layersModels.get( minLayerId );
				}
				else
				{
					innerMapping = new HashMap<Integer, ModelSpec>();
					layersModels.put( minLayerId, innerMapping );
				}
				// Assuming that no two files have the same model spec url values
				innerMapping.put( maxLayerId,  mspec );
			}
			catch (RuntimeException e)
			{
				System.err.println( "Error while reading file: " + fileUrl );
				e.printStackTrace( System.err );
				throw e;
			}
		}
		
		return layersModels;
	}

	private static AbstractModel< ? > createLayerModel( int modelIndex, mpicbg.models.CoordinateTransform modelTransformation )
	{
		AbstractModel< ? > res = null;

		switch ( modelIndex )
		{
		case 0:
			res = new TranslationModel2D();
			( ( TranslationModel2D )res ).set( ( TranslationModel2D ) modelTransformation );
			break;
		case 1:
			res = new RigidModel2D();
			( ( RigidModel2D )res ).set( ( RigidModel2D ) modelTransformation );
			break;
		case 2:
			res = new SimilarityModel2D();
			( ( SimilarityModel2D )res ).set( ( SimilarityModel2D ) modelTransformation );
			break;
		case 3:
			res = new AffineModel2D();
			( ( AffineModel2D )res ).set( ( AffineModel2D ) modelTransformation );
			break;
		case 4:
			res = new HomographyModel2D();
			( ( HomographyModel2D )res ).set( ( HomographyModel2D ) modelTransformation );
			break;
		default:
			throw new RuntimeException( "Unknown desired model" );
		}
		
		return res;
	}
	
	private static ArrayList< Tile< ? > > createLayersModels( int layersNum, int desiredModelIndex )
	{
		System.out.println( "Creating default models for each layer" );

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
			default:
				throw new RuntimeException( "Unknown desired model" );
			}
		}
		
		return tiles;
	}

	private static CoordinateTransform switchTransformType( AbstractAffineModel2D< ? > currentModel, int desiredModelIndex )
	{
		CoordinateTransform res = null;
		
		/* create tiles and models for all layers */
		switch ( desiredModelIndex )
		{
		case 0:
			res = new mpicbg.trakem2.transform.TranslationModel2D();
			( ( TranslationModel2D )res ).set( ( TranslationModel2D ) currentModel );
			break;
		case 1:
			res = new mpicbg.trakem2.transform.RigidModel2D();
			( ( RigidModel2D )res ).set( ( RigidModel2D ) currentModel );
			break;
		case 2:
			res = new mpicbg.trakem2.transform.SimilarityModel2D();
			( ( SimilarityModel2D )res ).set( ( SimilarityModel2D ) currentModel );
			break;
		case 3:
			res = new mpicbg.trakem2.transform.AffineModel2D();
			( ( AffineModel2D )res ).set( ( AffineModel2D ) currentModel );
			break;
		default:
			throw new RuntimeException( "Unknown desired model" );
		}
		
		return res;
	}


	
	private static void matchLayers(
			final ArrayList< Tile< ? > > tiles,
			final TileConfiguration tileConfiguration,
			final Map< Integer, Map< Integer, CorrespondenceSpec > > layersCorrs,
			final Map< Integer, Map< Integer, ModelSpec > > layersModels,
			final int ransacModelIndex,
			final float ransacMaxEpsilon,
			final List< Integer > fixedLayers,
			final int startLayer,
			final int endLayer,
			final Set<Integer> skippedLayers,
			final int maxDistance,
			final HashMap< Integer, List< Integer > > manuallyMatchedLayers )
	{
		
		System.out.println( "Matching layers" );

		for ( int layerA = startLayer; layerA < endLayer; layerA++ )
		{
			//if ( skippedLayers.contains( layerA ) || !layersCorrs.containsKey( layerA ) )
			if ( skippedLayers.contains( layerA ) )
			{
				System.out.println( "Skipping optimization of layer " + layerA );
				continue;
			}
			for ( Integer layerB : layersCorrs.get( layerA ).keySet() )
			//for ( int layerB = layerA + 1; layerB <= endLayer; layerB++ )
			//for ( int layerB = layerA + 1; layerB <= layerA + maxDistance; layerB++ )
			{
				// We compare both directions, so just do forward matching
				if ( layerB < layerA )
					continue;

				if ( layerB > endLayer )
					continue;
				
				if ( skippedLayers.contains( layerB ) )
				{
					System.out.println( "Skipping optimization of layer " + layerB );
					continue;
				}

				final boolean layer1Fixed = fixedLayers.contains( layerA );
				final boolean layer2Fixed = fixedLayers.contains( layerB );

				final CorrespondenceSpec corrspec12;
				final List< PointMatch > pm12;

				if ( !layersCorrs.containsKey( layerA ) || !layersCorrs.get( layerA ).containsKey( layerB ) )
				{
					corrspec12 = null;
					pm12 = null;
				}
				else
				{
					corrspec12 = layersCorrs.get( layerA ).get( layerB );
					pm12 = corrspec12.correspondencePointPairs;
				}

				// Check if there are corresponding layers to this layer, otherwise skip
				if ( pm12 == null )
					continue;

				
				
//				System.out.println( "Comparing layer " + layerA + " (fixed=" + layer1Fixed + ") to layer " +
//						layerB + " (fixed=" + layer2Fixed + ")" );
				
				if ( ( layer1Fixed && layer2Fixed ) )
					continue;


				// TODO: Load point matches
				
				final Tile< ? > t1 = tiles.get( layerA - startLayer );
				final Tile< ? > t2 = tiles.get( layerB - startLayer );

				if ( ( pm12 != null ) && ( pm12.size() > 1 ) )
				{
					// Apply the model to the points, and retrieve only the inliers
					AbstractModel< ? > model = createLayerModel( ransacModelIndex, layersModels.get( layerA ).get( layerB ).model.createTransform() );
					
					ArrayList< PointMatch > pm12Filtered = new ArrayList< PointMatch >();
					model.test( pm12, pm12Filtered, ransacMaxEpsilon, 0 );
					ArrayList< PointMatch > pm12FilteredReversed = new ArrayList< PointMatch >();
					for ( PointMatch pm : pm12Filtered )
					{
						PointMatch pmReversed = new PointMatch( pm.getP2(), pm.getP1() );
						pm12FilteredReversed.add( pmReversed );
					}
					/*
					 * adding Tiles to the initialing TileConfiguration, adding a Tile
					 * multiple times does not harm because the TileConfiguration is
					 * backed by a Set. 
					 */
					tileConfiguration.addTile( t1 );
					tileConfiguration.addTile( t2 );
					t1.connect( t2, pm12FilteredReversed );
					System.out.println( layerA + " <> " + layerB + " using " + pm12FilteredReversed.size() + " matches" );
				}

				if ( layer1Fixed )
					tileConfiguration.fixTile( t1 );

				if ( layer2Fixed )
					tileConfiguration.fixTile( t2 );

			}
			
		}

	}
	
	
	/**
	 * Optimizes the layers using elastic transformation,
	 * and updates the transformations of the tile-specs in the given layerTs.
	 * 
	 * @param param
	 * @param layersTs
	 * @param layersCorrs
	 * @param fixedLayers
	 * @param startLayer
	 * @param endLayer
	 * @param startX
	 * @param startY
	 */
	private static void optimizeElastic(
			final Params param,
			final HashMap< Integer, List< TileSpec > > layersTs,
			final Map< Integer, Map< Integer, CorrespondenceSpec > > layersCorrs,
			final Map< Integer, Map< Integer, ModelSpec > > layersModels,
			final List< Integer > fixedLayers,
			final int startLayer,
			final int endLayer,
			final int startX,
			final int startY,
			final Set<Integer> skippedLayers,
			final int maxDistance,
			final HashMap< Integer, List< Integer > > manuallyMatchedLayers )
	{
		final ArrayList< Tile< ? > > tiles = createLayersModels( endLayer - startLayer + 1, param.modelIndex );
		
		
		/* Initialization */
		final TileConfiguration tc = new TileConfiguration();
		tc.setThreadsNum( param.numThreads );
				
		matchLayers( tiles, tc,
				layersCorrs, layersModels,
				param.ransacModelIndex, param.ransacMaxEpsilon,
				fixedLayers, startLayer, endLayer,
				skippedLayers, maxDistance, manuallyMatchedLayers );
		
        /* Optimization */
		/*
        final TileConfiguration tileConfiguration = new TileConfiguration();

        for ( final Triple< Integer, Integer, Collection< PointMatch > > pair : pairs )
        {
                final Tile< ? > t1 = tiles.get( pair.a );
                final Tile< ? > t2 = tiles.get( pair.b );

                tileConfiguration.addTile( t1 );
                tileConfiguration.addTile( t2 );
                t2.connect( t1, pair.c );
        }

        for ( int i = 0; i < layerRange.size(); ++i )
        {
                final Layer layer = layerRange.get( i );
                if ( fixedLayers.contains( layer ) )
                        tileConfiguration.fixTile( tiles.get( i ) );
        }
*/


		try
		{
			final List< Tile< ? >  > nonPreAlignedTiles = tc.preAlign();


			System.out.println( "pre-aligned all but " + nonPreAlignedTiles.size() + " tiles" );

			tc.optimize(
					param.maxEpsilon,
					param.maxIterations,
					param.maxPlateauwidth );

		}
		catch ( Exception e )
		{
			throw new RuntimeException( e );
		}


		// Iterate the layers, and add the affine transform for each tile
		for ( int i = startLayer; i <= endLayer; ++i )
		{
			if ( skippedLayers.contains( i ) )
			{
				System.out.println( "Skipping saving after optimization of layer " + i );
				continue;
			}
			
			final List< TileSpec > layer = layersTs.get( i );
			final CoordinateTransform layerOutTransform = switchTransformType( ( AbstractAffineModel2D< ? > )tiles.get( i - startLayer ).getModel(), param.modelIndex );
			
			System.out.println( "Updating tiles in layer " + i );
			
			for ( TileSpec ts : layer )
			{
			    Transform addedTransform = new Transform();				    
			    addedTransform.className = layerOutTransform.getClass().getCanonicalName().toString();
			    addedTransform.dataString = layerOutTransform.toDataString();
			    
				ArrayList< Transform > outTransforms = new ArrayList< Transform >(Arrays.asList(ts.transforms));
				outTransforms.add(addedTransform);
				ts.transforms = outTransforms.toArray(ts.transforms);

				// bounding box after transformations are applied [left, right, top, bottom] possibly with extra entries for [front, back, etc.]
				final float[] boxMin = new float[ 2 ];
				final float[] boxMax = new float[ 2 ];
				( ( AbstractAffineModel2D< ? > )layerOutTransform ).estimateBounds( boxMin, boxMax );
				ts.bbox = new float[] { boxMin[0], boxMax[0], boxMin[1], boxMax[1] };
			}

			
		}


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
		
		Set<Integer> skippedLayers = Utils.parseRange( params.skippedLayers );
		
		List< String > actualTileSpecFiles;
		if ( params.tileSpecFiles.size() == 1 )
			// It might be a non-json file that contains a list of
			actualTileSpecFiles = Utils.getListFromFile( params.tileSpecFiles.get( 0 ) );
		else
			actualTileSpecFiles = params.tileSpecFiles;
		
		System.out.println( "Reading tilespecs" );

		// Load and parse tile spec files
		List< String > relevantTileSpecFiles = new ArrayList<String>();
		final HashMap< Integer, List< TileSpec > > layersTs = new HashMap<Integer, List<TileSpec>>();
		final HashMap< String, Integer > tsUrlToLayerIds = new HashMap<String, Integer>();
		final HashMap< Integer, String > layerIdToTsUrl = new HashMap<Integer, String>();
		for ( final String tsUrl : actualTileSpecFiles )
		{
			final TileSpec[] tileSpecs = TileSpecUtils.readTileSpecFile( tsUrl );
			int layer = tileSpecs[0].layer;
			if ( layer == -1 )
				throw new RuntimeException( "Error: a tile spec json file (" + tsUrl + ") has a tilespec without a layer " );
			
			if ( skippedLayers.contains( layer ) ) // No need to add skipped layers
				continue;

			relevantTileSpecFiles.add( tsUrl );
			layersTs.put( layer, Arrays.asList( tileSpecs ) );
			tsUrlToLayerIds.put( tsUrl, layer );
			layerIdToTsUrl.put( layer, tsUrl );
		}

		List< String > actualCorrFiles;
		if ( params.corrFiles.size() == 1 )
			// It might be a non-json file that contains a list of
			actualCorrFiles = Utils.getListFromFile( params.corrFiles.get( 0 ) );
		else
			actualCorrFiles = params.corrFiles;

		// Load and parse correspondence spec files
		final Map< Integer, Map< Integer, CorrespondenceSpec > > layersCorrs;
		layersCorrs = parseCorrespondenceFiles( actualCorrFiles, tsUrlToLayerIds, params.numThreads );

		List< String > actualModelFiles;
		if ( params.modelFiles.size() == 1 )
			// It might be a non-json file that contains a list of
			actualModelFiles = Utils.getListFromFile( params.modelFiles.get( 0 ) );
		else
			actualModelFiles = params.modelFiles;

		// Load and parse correspondence spec files
		final Map< Integer, Map< Integer, ModelSpec > > layersModels;
		layersModels = parseModelFiles( actualModelFiles, tsUrlToLayerIds );

		// Find bounding box
		System.out.println( "Finding bounding box" );
		final TileSpecsImage entireImage = TileSpecsImage.createImageFromFiles( relevantTileSpecFiles );
		final BoundingBox bbox = entireImage.getBoundingBox();
		
		int firstLayer = bbox.getStartPoint().getZ();
		if (( params.fromLayer != -1 ) && ( params.fromLayer >= firstLayer ))
			firstLayer = params.fromLayer;
		int lastLayer = bbox.getEndPoint().getZ();
		if (( params.toLayer != -1 ) && ( params.toLayer <= lastLayer ))
			lastLayer = params.toLayer;
		
		// Remove non existent fixed layers
		Iterator< Integer > fixedIt = params.fixedLayers.iterator();
		while ( fixedIt.hasNext() ) {
			int fixedLayer = fixedIt.next();
			if ( ( fixedLayer < firstLayer ) ||
				 ( fixedLayer > lastLayer ) ||
				 ( skippedLayers.contains( fixedLayer ) ) ) {
				fixedIt.remove();
			}
		}
		if ( params.fixedLayers.size() == 0 ) {
			params.fixedLayers.add( firstLayer );
		}
		
		// Parse manually matched layers
		HashMap< Integer, List< Integer > > manuallyMatchedLayers = new HashMap<Integer, List< Integer > >();
		if ( params.manualMatches.size() > 0 ) {
			for ( String pair : params.manualMatches ) {
				String[] vals = pair.split(":");
                if ( vals.length != 2 )
                        throw new IllegalArgumentException("Index pair not in correct format:" + pair);
                int layer1 = Integer.parseInt(vals[0]);
                int layer2 = Integer.parseInt(vals[1]);
				int minLayer = Math.min( layer1, layer2 );
				int maxLayer = Math.max( layer1, layer2 );
				if (! manuallyMatchedLayers.containsKey( minLayer ) ) {
					manuallyMatchedLayers.put( minLayer, new ArrayList< Integer >() );
				}
				List< Integer > pairsList = manuallyMatchedLayers.get( minLayer );
				pairsList.add( maxLayer );
			}
		}
		
		// Optimze
		optimizeElastic(
			params, layersTs, layersCorrs, layersModels,
			params.fixedLayers,
			firstLayer, lastLayer,
			bbox.getStartPoint().getX(), bbox.getStartPoint().getY(),
			skippedLayers,
			params.maxLayersDistance,
			manuallyMatchedLayers );

		// Save new tilespecs
		System.out.println( "Optimization complete. Generating tile transforms.");

		// Iterate through the layers and output the new tile specs
		for ( int layer = firstLayer; layer <= lastLayer; layer++ )
		{
			if ( skippedLayers.contains( layer ) )
				continue;
			
			String jsonFilename = layerIdToTsUrl.get( layer );
			String baseFilename = jsonFilename.substring( jsonFilename.lastIndexOf( '/' ) );
			
			String layerString = String.format( "%04d", layer );
			System.out.println( "Writing layer " + layerString );
			final File targetFile = new File( params.targetDir, baseFilename );
			final List< TileSpec > layerOutTiles = layersTs.get( layer );
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

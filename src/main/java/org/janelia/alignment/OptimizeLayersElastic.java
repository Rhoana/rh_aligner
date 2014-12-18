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
import java.util.List;

import mpicbg.models.AffineModel2D;
import mpicbg.models.HomographyModel2D;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.RigidModel2D;
import mpicbg.models.SimilarityModel2D;
import mpicbg.models.Spring;
import mpicbg.models.SpringMesh;
import mpicbg.models.Tile;
import mpicbg.models.TranslationModel2D;
import mpicbg.models.Vertex;
import mpicbg.trakem2.transform.MovingLeastSquaresTransform2;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.JsonSyntaxException;

public class OptimizeLayersElastic {
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--corrFiles", description = "Correspondence json files  (space separated) or a single file containing a line-separated list of json files", variableArity = true, required = true )
        public List<String> corrFiles = new ArrayList<String>();
        
        @Parameter( names = "--tilespecFiles", description = "Tilespec json files  (space separated) or a single file containing a line-separated list of json files", variableArity = true, required = true )
        public List<String> tileSpecFiles = new ArrayList<String>();
        
        @Parameter( names = "--fixedLayers", description = "Fixed layer numbers (space separated)", variableArity = true, required = true )
        public List<Integer> fixedLayers = new ArrayList<Integer>();
        
        @Parameter( names = "--imageWidth", description = "The width of the entire image (all layers), for consistent mesh computation", required = true )
        private int imageWidth;

        @Parameter( names = "--imageHeight", description = "The height of the entire image (all layers), for consistent mesh computation", required = true )
        private int imageHeight;

        @Parameter( names = "--targetDir", description = "Directory to output the new tilespec files", required = true )
        public String targetDir;

        @Parameter( names = "--modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int modelIndex = 1;
        
        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        private float layerScale = 0.1f;
        
        @Parameter( names = "--resolutionSpringMesh", description = "resolutionSpringMesh", required = false )
        private int resolutionSpringMesh = 32;
        
        //@Parameter( names = "--springLengthSpringMesh", description = "springLengthSpringMesh", required = false )
        //private float springLengthSpringMesh = 100f;
		
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
        
        //@Parameter( names = "--resolutionOutput", description = "resolutionOutput", required = false )
        //private int resolutionOutput = 128;
        
        @Parameter( names = "--useLegacyOptimizer", description = "Use legacy optimizer", required = false )
        private boolean useLegacyOptimizer = false;

        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
        @Parameter( names = "--fromLayer", description = "The layer to start the optimization from (default: first layer in the tile specs data)", required = false )
        private int fromLayer = -1;

        @Parameter( names = "--toLayer", description = "The last layer to include in the optimization (default: last layer in the tile specs data)", required = false )
        private int toLayer = -1;
               
        @Parameter( names = "--skipLayers", description = "The layers ranges that will not be processed (default: none)", required = false )
        private String skippedLayers = "";

	}
	
	private OptimizeLayersElastic() {}
	
	private static HashMap< Integer, HashMap< Integer, CorrespondenceSpec > > parseCorrespondenceFiles(
			final List< String > fileUrls,
			final HashMap< String, Integer > tsUrlToLayerIds )
	{
		HashMap< Integer, HashMap< Integer, CorrespondenceSpec > > layersCorrs = new HashMap<Integer, HashMap<Integer,CorrespondenceSpec>>();
		
		for ( String fileUrl : fileUrls )
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

			for ( final CorrespondenceSpec corr : corr_data )
			{
				final int layer1Id = tsUrlToLayerIds.get( corr.url1 );
				final int layer2Id = tsUrlToLayerIds.get( corr.url2 );
				final HashMap< Integer, CorrespondenceSpec > innerMapping;

				if ( layersCorrs.containsKey( layer1Id ) )
				{
					innerMapping = layersCorrs.get( layer1Id );
				}
				else
				{
					innerMapping = new HashMap<Integer, CorrespondenceSpec>();
					layersCorrs.put( layer1Id, innerMapping );
				}
				// Assuming that no two files have the same correspondence spec url values
				innerMapping.put( layer2Id,  corr );
			}
		}
		
		return layersCorrs;
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

	private static boolean compareArrays( float[] a, float[] b )
	{
		if ( a.length != b.length )
			return false;
		
		for ( int i = 0; i < a.length; i++ )
			// if ( a[i] != b[i] )
			if ( Math.abs( a[i] - b[i] ) > 2 * Math.ulp( b[i] ) )
				return false;
		
		return true;
	}
	
	/* Fixes the point match P1 vertices to point to the given vertices (same objects) */
	private static List< PointMatch > fixPointMatchVertices(
			List< PointMatch > pms,
			ArrayList< Vertex > vertices )
	{
		List< PointMatch > newPms = new ArrayList<PointMatch>( pms.size() );
		
		for ( final PointMatch pm : pms )
		{
			// Search for the given point match p1 point in the vertices list,
			// and if found, link the vertex instead of that point
			for ( final Vertex v : vertices )
			{
				if ( compareArrays( pm.getP1().getL(), v.getL() )  )
				{
					// Copy the new world values, in case there was a slight drift 
					for ( int i = 0; i < v.getW().length; i++ )
						v.getW()[ i ] = pm.getP1().getW()[ i ];
					
					PointMatch newPm = new PointMatch( v, pm.getP2(), pm.getWeights() );
					newPms.add( newPm );
				}
			}
		}
		
		return newPms;
	}

	private static ArrayList< SpringMesh > fixAllPointMatchVertices(
			final Params param,
			final HashMap< Integer, HashMap< Integer, CorrespondenceSpec > > layersCorrs,
			final int startLayer,
			final int endLayer )
	{
		final int meshWidth = ( int )Math.ceil( param.imageWidth * param.layerScale );
		final int meshHeight = ( int )Math.ceil( param.imageHeight * param.layerScale );
		
		final ArrayList< SpringMesh > meshes = new ArrayList< SpringMesh >( endLayer - startLayer + 1 );
		for ( int i = startLayer; i <= endLayer; ++i )
		{
			final SpringMesh singleMesh = new SpringMesh(
					param.resolutionSpringMesh,
					meshWidth,
					meshHeight,
					param.stiffnessSpringMesh,
					param.maxStretchSpringMesh * param.layerScale,
					param.dampSpringMesh ); 
			meshes.add( singleMesh );
			
			if ( layersCorrs.containsKey( i ) )
			{
				HashMap< Integer, CorrespondenceSpec > layerICorrs = layersCorrs.get( i );
				for ( CorrespondenceSpec corrspec : layerICorrs.values() )
				{
					final List< PointMatch > pms = corrspec.correspondencePointPairs;
					if ( pms != null )
					{
						final List< PointMatch > pmsFixed = fixPointMatchVertices( pms, singleMesh.getVertices() );
						corrspec.correspondencePointPairs = pmsFixed;
					}
					
				}
			}
		}

		return meshes;
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
			final HashMap< Integer, HashMap< Integer, CorrespondenceSpec > > layersCorrs,
			final List< Integer > fixedLayers,
			final int startLayer,
			final int endLayer,
			final int startX,
			final int startY,
			final List<Integer> skippedLayers )
	{
		final ArrayList< Tile< ? > > tiles = createLayersModels( endLayer - startLayer + 1, param.modelIndex );
		
		/* Initialization */
		final TileConfiguration initMeshes = new TileConfiguration();
		initMeshes.setThreadsNum( param.numThreads );
				
		final ArrayList< SpringMesh > meshes = fixAllPointMatchVertices(
				param, layersCorrs, startLayer, endLayer );
		
		for ( int layerA = startLayer; layerA < endLayer; layerA++ )
		{
			if ( skippedLayers.contains( layerA ) )
			{
				System.out.println( "Skipping optimization of layer " + layerA );
				continue;
			}
			for ( int layerB = layerA + 1; layerB <= endLayer; layerB++ )
			{

				if ( skippedLayers.contains( layerB ) )
				{
					System.out.println( "Skipping optimization of layer " + layerB );
					continue;
				}

				final boolean layer1Fixed = fixedLayers.contains( layerA );
				final boolean layer2Fixed = fixedLayers.contains( layerB );

				final CorrespondenceSpec corrspec12;
				final List< PointMatch > pm12;
				final CorrespondenceSpec corrspec21;
				final List< PointMatch > pm21;

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

				if ( !layersCorrs.containsKey( layerB ) || !layersCorrs.get( layerB ).containsKey( layerA ) )
				{
					corrspec21 = null;
					pm21 = null;
				}
				else
				{
					corrspec21 = layersCorrs.get( layerB ).get( layerA );
					pm21 = corrspec21.correspondencePointPairs;
				}

				// Check if there are corresponding layers to this layer, otherwise skip
				if ( pm12 == null && pm21 == null )
					continue;

				
				
//				System.out.println( "Comparing layer " + layerA + " (fixed=" + layer1Fixed + ") to layer " +
//						layerB + " (fixed=" + layer2Fixed + ")" );
				
				if ( ( layer1Fixed && layer2Fixed ) )
					continue;

				final SpringMesh m1 = meshes.get( layerA - startLayer );
				final SpringMesh m2 = meshes.get( layerB - startLayer );

				// TODO: Load point matches
				
				final Tile< ? > t1 = tiles.get( layerA - startLayer );
				final Tile< ? > t2 = tiles.get( layerB - startLayer );

				final float springConstant  = 1.0f / ( layerB - layerA );
				

				if ( layer1Fixed )
					initMeshes.fixTile( t1 );
				else
				{
					if ( ( pm12 != null ) && ( pm12.size() > 1 ) )
					{
						//final List< PointMatch > pm12Fixed = fixPointMatchVertices( pm12, m1.getVertices() );
						
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
						if ( corrspec12.shouldConnect )
						{
							initMeshes.addTile( t1 );
							initMeshes.addTile( t2 );
							t1.connect( t2, pm12 );
						}

					}

				}

				if ( layer2Fixed )
					initMeshes.fixTile( t2 );
				else
				{
					if ( ( pm21 != null ) && ( pm21.size() > 1 ) )
					{
						//final List< PointMatch > pm21Fixed = fixPointMatchVertices( pm21, m2.getVertices() );

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
						if ( corrspec21.shouldConnect )
						{
							initMeshes.addTile( t1 );
							initMeshes.addTile( t2 );
							t2.connect( t1, pm21 );
						}
					}

				}
			
				System.out.println( layerA + " <> " + layerB + " spring constant = " + springConstant );

			}
			
		}

		
		
		
		
//		
//		
//		final Set< Integer > layersIds = layersTs.keySet();
//		for ( final Integer layerId : layersIds )
//		{
//			// Check if there are corresponding layers to this layer, otherwise skip
//			if ( !layersCorrs.containsKey( layerId ) )
//				continue;
//			
//			final Set< Integer > corrLayerIds = layersCorrs.get( layerId ).keySet();
//			for ( final Integer corrLayerId : corrLayerIds )
//			{
//				// Only looking for forward comparisons
//				if ( corrLayerId.intValue() < layerId.intValue() )
//				{
//					System.out.println( "Skipping on comparing layers: " + layerId + " to " + corrLayerId );
//					continue;
//				}
//				
//				final boolean layer1Fixed = fixedLayers.contains( layerId );
//				final boolean layer2Fixed = fixedLayers.contains( corrLayerId );
//
//				System.out.println( "Comparing layer " + layerId + " (fixed=" + layer1Fixed + ") to layer " +
//						corrLayerId + " (fixed=" + layer2Fixed + ")" );
//				
//				if ( ( layer1Fixed && layer2Fixed ) )
//					continue;
//
//				final SpringMesh m1 = meshes.get( layerId - startLayer );
//				final SpringMesh m2 = meshes.get( corrLayerId - startLayer );
//
//				// TODO: Load point matches
//				
//				final Tile< ? > t1 = tiles.get( layerId - startLayer );
//				final Tile< ? > t2 = tiles.get( corrLayerId - startLayer );
//
//				final float springConstant  = 1.0f / ( corrLayerId - layerId );
//
//				if ( layer1Fixed )
//					initMeshes.fixTile( t1 );
//				else
//				{
//					final CorrespondenceSpec corrspec12 = layersCorrs.get( layerId ).get( corrLayerId );
//					final List< PointMatch > pm12 = corrspec12.correspondencePointPairs;
//					
//					for ( final PointMatch pm : pm12 )
//					{
//						final Vertex p1 = new Vertex( pm.getP1() );
//						final Vertex p2 = new Vertex( pm.getP2() );
//						p1.addSpring( p2, new Spring( 0, springConstant ) );
//						m2.addPassiveVertex( p2 );
//					}
//					
//					/*
//					 * adding Tiles to the initialing TileConfiguration, adding a Tile
//					 * multiple times does not harm because the TileConfiguration is
//					 * backed by a Set. 
//					 */
//					if ( corrspec12.shouldConnect )
//					{
//						initMeshes.addTile( t1 );
//						initMeshes.addTile( t2 );
//						t1.connect( t2, pm12 );
//					}
//
//				}
//
//				if ( layer2Fixed )
//					initMeshes.fixTile( t2 );
//				else
//				{
//					final CorrespondenceSpec corrspec21 = layersCorrs.get( corrLayerId ).get( layerId );
//					final List< PointMatch > pm21 = corrspec21.correspondencePointPairs;
//
//					for ( final PointMatch pm : pm21 )
//					{
//						final Vertex p1 = new Vertex( pm.getP1() );
//						final Vertex p2 = new Vertex( pm.getP2() );
//						p1.addSpring( p2, new Spring( 0, springConstant ) );
//						m1.addPassiveVertex( p2 );
//					}
//					
//					/*
//					 * adding Tiles to the initialing TileConfiguration, adding a Tile
//					 * multiple times does not harm because the TileConfiguration is
//					 * backed by a Set. 
//					 */
//					if ( corrspec21.shouldConnect )
//					{
//						initMeshes.addTile( t1 );
//						initMeshes.addTile( t2 );
//						t2.connect( t1, pm21 );
//					}
//				}
//
//				System.out.println( layerId + " <> " + corrLayerId + " spring constant = " + springConstant );
//			}
//		}
		
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
		
		// Find current bounding box of tilespecs
		
		
		/* translate relative to bounding box */
		final int boxX = startX;
		final int boxY = startY;
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

		
		// Iterate the layers, and add the mesh transform for each tile
		for ( int i = startLayer; i <= endLayer; ++i )
		{
			if ( skippedLayers.contains( i ) )
			{
				System.out.println( "Skipping saving after optimization of layer " + i );
				continue;
			}
			
			final SpringMesh mesh = meshes.get( i - startLayer );
			final List< TileSpec > layer = layersTs.get( i );
			
			System.out.println( "Updating tiles in layer " + i );
			
			for ( TileSpec ts : layer )
			{
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
					final MovingLeastSquaresTransform2 mlt = new MovingLeastSquaresTransform2();
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
					System.out.println( "Error applying transform to tile in layer " + i + "." );
					e.printStackTrace();
				}

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
		
		List< String > actualTileSpecFiles;
		if ( params.tileSpecFiles.size() == 1 )
			// It might be a non-json file that contains a list of
			actualTileSpecFiles = Utils.getListFromFile( params.tileSpecFiles.get( 0 ) );
		else
			actualTileSpecFiles = params.tileSpecFiles;
		
		// Load and parse tile spec files
		final HashMap< Integer, List< TileSpec > > layersTs = new HashMap<Integer, List<TileSpec>>();
		final HashMap< String, Integer > tsUrlToLayerIds = new HashMap<String, Integer>();
		final HashMap< Integer, String > layerIdToTsUrl = new HashMap<Integer, String>();
		for ( final String tsUrl : actualTileSpecFiles )
		{
			final TileSpec[] tileSpecs = TileSpecUtils.readTileSpecFile( tsUrl );
			int layer = tileSpecs[0].layer;
			if ( layer == -1 )
				throw new RuntimeException( "Error: a tile spec json file (" + tsUrl + ") has a tilespec without a layer " );

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
		final HashMap< Integer, HashMap< Integer, CorrespondenceSpec > > layersCorrs;
		layersCorrs = parseCorrespondenceFiles( actualCorrFiles, tsUrlToLayerIds );

		// Find bounding box
		final TileSpecsImage entireImage = TileSpecsImage.createImageFromFiles( actualTileSpecFiles );
		final BoundingBox bbox = entireImage.getBoundingBox();
		
		int firstLayer = bbox.getStartPoint().getZ();
		if (( params.fromLayer != -1 ) && ( params.fromLayer >= firstLayer ))
			firstLayer = params.fromLayer;
		int lastLayer = bbox.getEndPoint().getZ();
		if (( params.toLayer != -1 ) && ( params.toLayer <= lastLayer ))
			lastLayer = params.toLayer;
		
		List<Integer> skippedLayers = Utils.parseRange( params.skippedLayers );
		
		// Optimze
		optimizeElastic(
			params, layersTs, layersCorrs,
			params.fixedLayers,
			firstLayer, lastLayer,
			bbox.getStartPoint().getX(), bbox.getStartPoint().getY(),
			skippedLayers );

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

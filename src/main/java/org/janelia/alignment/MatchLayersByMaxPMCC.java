package org.janelia.alignment;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;

import java.awt.geom.AffineTransform;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.Writer;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutionException;

import mpicbg.ij.blockmatching.BlockMatching;
import mpicbg.models.AbstractAffineModel2D;
import mpicbg.models.AbstractModel;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.ErrorStatistic;
import mpicbg.models.IdentityModel;
import mpicbg.models.InvertibleCoordinateTransform;
import mpicbg.models.NoninvertibleModelException;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.RigidModel2D;
import mpicbg.models.SpringMesh;
import mpicbg.models.TranslationModel2D;
import mpicbg.models.Vertex;
import mpicbg.trakem2.align.Util;
import mpicbg.trakem2.transform.AffineModel2D;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

public class MatchLayersByMaxPMCC {
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--inputfile1", description = "The first layer tilespec file", required = true )
        private String inputfile1;

        @Parameter( names = "--inputfile2", description = "The second layer tilespec file", required = true )
        private String inputfile2;

        @Parameter( names = "--modelsfile1", description = "The models from the first layer file", required = true )
        private String modelsfile1;

        @Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;

        @Parameter( names = "--imageWidth", description = "The width of the entire image (all layers), for consistent mesh computation", required = true )
        private int imageWidth;

        @Parameter( names = "--imageHeight", description = "The height of the entire image (all layers), for consistent mesh computation", required = true )
        private int imageHeight;

        @Parameter( names = "--meshesDir1", description = "The directory where the cached mesh per tile of the first image is located", required = false )
        private String meshesDir1 = null;

        @Parameter( names = "--meshesDir2", description = "The directory where the cached mesh per tile of the second image is located", required = false )
        private String meshesDir2 = null;

        @Parameter( names = "--autoAddModel", description = "Automatically add the Identity model in case a model is not found", required = false )
        private boolean autoAddModel = false;

        @Parameter( names = "--fixedLayers", description = "Fixed layer numbers (space separated)", variableArity = true, required = false )
        public List<Integer> fixedLayers = new ArrayList<Integer>();
        
        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        public float layerScale = 0.1f;
        
        @Parameter( names = "--searchRadius", description = "Search window radius", required = false )
        public int searchRadius = 200;
        
        @Parameter( names = "--blockRadius", description = "Matching block radius", required = false )
        public int blockRadius = -1;
                
//        @Parameter( names = "--resolution", description = "Resolution", required = false )
//        public int resolution = 16;
        
        @Parameter( names = "--minR", description = "minR", required = false )
        public float minR = 0.6f;
        
        @Parameter( names = "--maxCurvatureR", description = "maxCurvatureR", required = false )
        public float maxCurvatureR = 10.0f;
        
        @Parameter( names = "--rodR", description = "rodR", required = false )
        public float rodR = 0.9f;
        
        @Parameter( names = "--useLocalSmoothnessFilter", description = "useLocalSmoothnessFilter", required = false )
        public boolean useLocalSmoothnessFilter = false;
        
        @Parameter( names = "--localModelIndex", description = "localModelIndex", required = false )
        public int localModelIndex = 1;
        // 0 = "Translation", 1 = "Rigid", 2 = "Similarity", 3 = "Affine"
        
        @Parameter( names = "--localRegionSigma", description = "localRegionSigma", required = false )
        public float localRegionSigma = 200f;
        
        @Parameter( names = "--maxLocalEpsilon", description = "maxLocalEpsilon", required = false )
        public float maxLocalEpsilon = 12f;
        
        @Parameter( names = "--maxLocalTrust", description = "maxLocalTrust", required = false )
        public int maxLocalTrust = 3;
        
        //@Parameter( names = "--maxNumNeighbors", description = "maxNumNeighbors", required = false )
        //public float maxNumNeighbors = 3f;
        		
        @Parameter( names = "--resolutionSpringMesh", description = "resolutionSpringMesh", required = false )
        private int resolutionSpringMesh = 32;
        
        @Parameter( names = "--stiffnessSpringMesh", description = "stiffnessSpringMesh", required = false )
        public float stiffnessSpringMesh = 0.1f;
		
        @Parameter( names = "--dampSpringMesh", description = "dampSpringMesh", required = false )
        public float dampSpringMesh = 0.9f;
		
        @Parameter( names = "--maxStretchSpringMesh", description = "maxStretchSpringMesh", required = false )
        public float maxStretchSpringMesh = 2000.0f;

        @Parameter( names = "--springLengthSpringMesh", description = "spring_length", required = false )
        public float springLengthSpringMesh = 100.0f;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
        @Parameter( names = "--rectsPerDim", description = "Number of rectangles per row/column in which intermediate matching results will be saved", required = false )
        public int rectsPerDim = 1;
                
	}
	
	private final static boolean PRINT_TIME_PER_STEP = true;
	
	private MatchLayersByMaxPMCC() {}

	private static String getIntermediateFileName( final String targetPath,
			final int rectRow,
			final int rectColumn )
	{
		String outputFile = targetPath.substring( 0, targetPath.lastIndexOf( '.' ) ) +
				"_r" + rectRow + "_c" + rectColumn + ".json";
		return outputFile;
	}

	private static CorrespondenceSpec[] loadIntermediateResults(
			final Params params,
			final int rectRow,
			final int rectColumn )
	{
		CorrespondenceSpec[] res = null;

		final String inputFileName = getIntermediateFileName( params. targetPath, rectRow, rectColumn );
		File inFile = new File( inputFileName );
		if ( inFile.exists() )
		{
			System.out.println( "Intermediate file: " + inputFileName + " exists, loading data" );
			// Open and parse the json file
			final CorrespondenceSpec[] corr_data;
			try
			{
				final Gson gson = new Gson();
				corr_data = gson.fromJson( new InputStreamReader( new FileInputStream( inFile ) ),
						CorrespondenceSpec[].class );
			}
			catch ( final JsonSyntaxException e )
			{
				System.err.println( "JSON syntax malformed." );
				e.printStackTrace( System.err );
				return null;
			}
			catch ( final Exception e )
			{
				e.printStackTrace( System.err );
				return null;
			}

			if ( corr_data != null )
			{
				// There should only be two correspondence points lists in the intermediate files
				assert( corr_data.length == 2 );

				res = corr_data;
			}

		}

		return res;
	}

	private static void saveIntermediateResults(
			final List< CorrespondenceSpec > corr_data,
			final Params params,
			final int rectRow,
			final int rectColumn )

	{
		try {
			String outputFile = getIntermediateFileName( params.targetPath, rectRow, rectColumn );
			System.out.println( "Saving intermediate result to: " + outputFile );
			Writer writer = new FileWriter(outputFile);
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
			final TileSpecsImage layerTileSpecsImage,
			final int layer,
			final FloatProcessor output,
			final FloatProcessor alpha,
			final int mipmapLevel,
			final float layerScale,
			final String meshesDir )
	{
		final ByteProcessor tp;
		if ( meshesDir == null )
			tp = layerTileSpecsImage.render( layer, mipmapLevel, layerScale );
		else
			tp = layerTileSpecsImage.renderFromMeshes( meshesDir, layer, mipmapLevel, layerScale );
		final byte[] inputPixels = ( byte[] )tp.getPixels();
		for ( int i = 0; i < inputPixels.length; ++i )
		{
			/*
			final int argb = inputPixels[ i ];
			final int a = ( argb >> 24 ) & 0xff;
			final int r = ( argb >> 16 ) & 0xff;
			final int g = ( argb >> 8 ) & 0xff;
			final int b = argb & 0xff;
			
			final float v = ( r + g + b ) / ( float )3;
			final float w = a / ( float )255;
			
			output.setf( i, v );
			alpha.setf( i, w );
			*/
			output.setf( i,  ( float )( inputPixels[ i ] / 255.0f ));
			alpha.setf( i,  1.0f );
		}
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
	private static FloatProcessor tilespecToFloatAndMask(
			final TileSpecsImage layerTileSpecsImage,
			final int layer,
			final int mipmapLevel,
			final float layerScale,
			final String meshesDir )
	{
		final ByteProcessor tp;
		if ( meshesDir == null )
			tp = layerTileSpecsImage.render( layer, mipmapLevel, layerScale );
		else
			tp = layerTileSpecsImage.renderFromMeshes( meshesDir, layer, mipmapLevel, layerScale );
		return tp.convertToFloatProcessor();
	}

	/**
	 * Receives a single layer, applies the transformations, and outputs the ip and mask
	 * of the given level (render the ip and ipMask), according to a given bounding box
	 * 
	 * @param layerTileSpecs
	 * @param ip
	 * @param ipMask
	 * @param mipmapLevel
	 */
	private static FloatProcessor tilespecToFloatAndMask(
			final TileSpecsImage layerTileSpecsImage,
			final int layer,
			final int mipmapLevel,
			final float layerScale,
			final String meshesDir,
			final BoundingBox bbox )
	{
		final ByteProcessor tp;
		if ( meshesDir == null )
			tp = layerTileSpecsImage.render( layer, mipmapLevel, layerScale, bbox.getWidth(), bbox.getHeight(), bbox.getStartPoint().getX(), bbox.getStartPoint().getY() );
		else
			throw new UnsupportedOperationException( "No support yet for rendering using meshes and bounding box" );
			//tp = layerTileSpecsImage.renderFromMeshes( meshesDir, layer, mipmapLevel, layerScale, bbox.getWidth(), bbox.getHeight(), bbox.getStartPoint().getX(), bbox.getStartPoint().getY() );
		return tp.convertToFloatProcessor();
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
	private static ByteProcessor tilespecToByteAndMask(
			final TileSpecsImage layerTileSpecsImage,
			final int layer,
			final int mipmapLevel,
			final float layerScale,
			final String meshesDir )
	{
		final ByteProcessor tp;
		if ( meshesDir == null )
			tp = layerTileSpecsImage.render( layer, mipmapLevel, layerScale );
		else
			tp = layerTileSpecsImage.renderFromMeshes( meshesDir, layer, mipmapLevel, layerScale );
		return tp;
	}

/*
	private static Tile< ? >[] createLayersModels( int layersNum, int desiredModelIndex )
	{
		/// create tiles and models for all layers
		final Tile< ? >[] tiles = new Tile< ? >[ layersNum ];
		for ( int i = 0; i < layersNum; ++i )
		{
			switch ( desiredModelIndex )
			{
			case 0:
				tiles[i] = new Tile< TranslationModel2D >( new TranslationModel2D() );
				break;
			case 1:
				tiles[i] = new Tile< RigidModel2D >( new RigidModel2D() );
				break;
			case 2:
				tiles[i] = new Tile< SimilarityModel2D >( new SimilarityModel2D() );
				break;
			case 3:
				tiles[i] = new Tile< AffineModel2D >( new AffineModel2D() );
				break;
			case 4:
				tiles[i] = new Tile< HomographyModel2D >( new HomographyModel2D() );
				break;
			default:
				throw new RuntimeException( "Unknown desired model" );
			}
		}
		
		return tiles;
	}
*/
	
	private static List< CorrespondenceSpec > matchLayersByMaxPMCC(
			final Params param,
			final AbstractModel< ? > model,
			final TileSpec[] ts1,
			final TileSpec[] ts2,
			int mipmapLevel )
	{
		final List< CorrespondenceSpec > corr_data = new ArrayList< CorrespondenceSpec >();

/*
		final TileConfiguration initMeshes = new TileConfiguration();
*/
		final int layer1 = ts1[0].layer;
		final int layer2 = ts2[0].layer;
		

		// Compute bounding box of the two layers
		
		
		long startTime = System.currentTimeMillis();
		// Create the meshes for the tiles
		/*
		final List< TileSpec[] > tsList = new ArrayList< TileSpec[] >();
		tsList.add( ts1 );
		tsList.add( ts2 );
		
		final List< SpringMesh > meshes = Utils.createMeshes(
				tsList, param.imageWidth, param.imageHeight,
				param.springLengthSpringMesh, param.stiffnessSpringMesh,
				param.maxStretchSpringMesh, param.layerScale, param.dampSpringMesh );
		*/
		
		final int meshWidth = ( int )Math.ceil( param.imageWidth * param.layerScale );
		final int meshHeight = ( int )Math.ceil( param.imageHeight * param.layerScale );
		final SpringMesh[] meshes = new SpringMesh[2];
		for ( int i = 0; i < meshes.length; i++ )
			meshes[i] = new SpringMesh(
					param.resolutionSpringMesh,
					meshWidth,
					meshHeight,
					param.stiffnessSpringMesh,
					param.maxStretchSpringMesh * param.layerScale,
					param.dampSpringMesh );
		
		long endTime = System.currentTimeMillis();
		if ( PRINT_TIME_PER_STEP )
			System.out.println("Creating mesh took: " + ((endTime - startTime) / 1000.0) + " sec");

		//final int blockRadius = Math.max( 32, meshWidth / p.resolutionSpringMesh / 2 );
//		final int param_blockRadius = param.imageWidth / param.resolution / 2;
		final int orig_param_resolution = 16;
		final int param_blockRadius;
		if ( param.blockRadius < 0 )
			param_blockRadius = param.imageWidth / orig_param_resolution / 2;
		else
			param_blockRadius = param.blockRadius;
		final int blockRadius = Math.max( 16, mpicbg.util.Util.roundPos( param.layerScale * param_blockRadius ) );
//		final int blockRadius = Math.max( mpicbg.util.Util.roundPos( 16 / param.layerScale ), param.blockRadius );
		
		System.out.println( "effective block radius = " + blockRadius );
		
		/* scale pixel distances */
		final int searchRadius = ( int )Math.round( param.layerScale * param.searchRadius );
//		final int searchRadius = param.searchRadius;
		final float localRegionSigma = param.layerScale * param.localRegionSigma;
		final float maxLocalEpsilon = param.layerScale * param.maxLocalEpsilon;
//		final float localRegionSigma = param.localRegionSigma;
//		final float maxLocalEpsilon = param.maxLocalEpsilon;
		
		startTime = System.currentTimeMillis();
		final AbstractModel< ? > localSmoothnessFilterModel = Util.createModel( param.localModelIndex );
		endTime = System.currentTimeMillis();
		if ( PRINT_TIME_PER_STEP )
			System.out.println("Creating model took: " + ((endTime - startTime) / 1000.0) + " sec");

		
		final SpringMesh m1 = meshes[ 0 ];
		final SpringMesh m2 = meshes[ 1 ];

		final ArrayList< PointMatch > pm12 = new ArrayList< PointMatch >();
		final ArrayList< PointMatch > pm21 = new ArrayList< PointMatch >();

		final ArrayList< Vertex > v1 = m1.getVertices();
		final ArrayList< Vertex > v2 = m2.getVertices();
		
				
		/* Load images and masks into FloatProcessor objects */		
		System.out.println( "Rendering layers" );
		startTime = System.currentTimeMillis();
		final TileSpecsImage layer1Img = new TileSpecsImage( ts1 );
		final TileSpecsImage layer2Img = new TileSpecsImage( ts2 );
		
		layer1Img.setThreadsNum( param.numThreads );
		layer2Img.setThreadsNum( param.numThreads );
		
		/*
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
		*/
		
		// TODO: load the tile specs to FloatProcessor objects
		final FloatProcessor ip1 = tilespecToFloatAndMask( layer1Img, layer1, mipmapLevel, param.layerScale, param.meshesDir1 );
		final FloatProcessor ip2 = tilespecToFloatAndMask( layer2Img, layer2, mipmapLevel, param.layerScale, param.meshesDir2 );
//		final ByteProcessor ip1 = tilespecToByteAndMask( layer1Img, layer1, mipmapLevel, param.layerScale, param.meshesDir1 );
//		final ByteProcessor ip2 = tilespecToByteAndMask( layer2Img, layer2, mipmapLevel, param.layerScale, param.meshesDir2 );
		endTime = System.currentTimeMillis();
		if ( PRINT_TIME_PER_STEP )
			System.out.println("Creating images took: " + ((endTime - startTime) / 1000.0) + " sec");
		
		//final float springConstant  = 1.0f / ( layer2 - layer1 );
		
		// Scale the affine transformation
		final AffineTransform scaleDown = new AffineTransform();
		scaleDown.scale( param.layerScale, param.layerScale );
		final AffineModel2D scaleDownModel = new AffineModel2D();
		scaleDownModel.set( scaleDown );
		

		final CoordinateTransformList< CoordinateTransform > scaledModel = new CoordinateTransformList< CoordinateTransform >();
		scaledModel.add( scaleDownModel.createInverse() );
		scaledModel.add( ( CoordinateTransform )model );
		scaledModel.add( scaleDownModel );
		
		final CoordinateTransformList< CoordinateTransform > scaledInverseModel = new CoordinateTransformList< CoordinateTransform >();
		scaledInverseModel.add( scaleDownModel.createInverse() );
		scaledInverseModel.add( ( ( InvertibleCoordinateTransform )model ).createInverse() );
		scaledInverseModel.add( scaleDownModel );

		
		try
		{
			startTime = System.currentTimeMillis();
			BlockMatching.matchByMaximalPMCC(
					ip1,
					ip2,
					null,//ip1Mask,
					null,//ip2Mask,
					1.0f,
					scaledInverseModel, //( ( InvertibleCoordinateTransform )model ).createInverse(),
					blockRadius,
					blockRadius,
					searchRadius,
					searchRadius,
					param.minR,
					param.rodR,
					param.maxCurvatureR,
					v1,
					pm12,
					new ErrorStatistic( 1 ) );
			endTime = System.currentTimeMillis();
			if ( PRINT_TIME_PER_STEP )
				System.out.println("Block matching 1 took: " + ((endTime - startTime) / 1000.0) + " sec");
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

		if ( param.useLocalSmoothnessFilter )
		{

			startTime = System.currentTimeMillis();
			System.out.println( layer1 + " > " + layer2 + ": found " + pm12.size() + " correspondence candidates." );
			localSmoothnessFilterModel.localSmoothnessFilter( pm12, pm12, localRegionSigma, maxLocalEpsilon, param.maxLocalTrust );
			System.out.println( layer1 + " > " + layer2 + ": " + pm12.size() + " candidates passed local smoothness filter." );
			endTime = System.currentTimeMillis();
			if ( PRINT_TIME_PER_STEP )
				System.out.println("local smooth filter 1 took: " + ((endTime - startTime) / 1000.0) + " sec");
		}
		else
		{
			System.out.println( layer1 + " > " + layer2 + ": found " + pm12.size() + " correspondences." );
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
/*				
				for ( final PointMatch pm : pm12 )
				{
					final Vertex p1 = ( Vertex )pm.getP1();
					final Vertex p2 = new Vertex( pm.getP2() );
					p1.addSpring( p2, new Spring( 0, springConstant ) );
					m2.addPassiveVertex( p2 );
				}
*/

		/*
		 * adding Tiles to the initialing TileConfiguration, adding a Tile
		 * multiple times does not harm because the TileConfiguration is
		 * backed by a Set. 
		 */
/*
				if ( pm12.size() > model.getMinNumMatches() )
				{
					initMeshes.addTile( t1 );
					initMeshes.addTile( t2 );
					t1.connect( t2, pm12 );
				}
*/
		
		// Remove Vertex (spring mesh) details from points
		final ArrayList< PointMatch > pm12_strip = new ArrayList< PointMatch >();
		for (PointMatch pm: pm12)
		{
			PointMatch actualPm;
			try {
				actualPm = new PointMatch(
						new Point( pm.getP1().getL(), scaleDownModel.applyInverse( pm.getP1().getW() ) ),
						new Point( pm.getP2().getL(), scaleDownModel.applyInverse( pm.getP2().getW() ) )
						);
				pm12_strip.add( actualPm );
			} catch (NoninvertibleModelException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		// TODO: Export / Import master sprint mesh vertices no calculated  individually per tile (v1, v2).
		corr_data.add(new CorrespondenceSpec(
				mipmapLevel,
				param.inputfile1,
				param.inputfile2,
				pm12_strip,
				( pm12.size() > model.getMinNumMatches() ) ));

		try
		{
			startTime = System.currentTimeMillis();
			BlockMatching.matchByMaximalPMCC(
					ip2,
					ip1,
					null,//ip2Mask,
					null,//ip1Mask,
					1.0f,
					scaledModel, //model,
					blockRadius,
					blockRadius,
					searchRadius,
					searchRadius,
					param.minR,
					param.rodR,
					param.maxCurvatureR,
					v2,
					pm21,
					new ErrorStatistic( 1 ) );
			endTime = System.currentTimeMillis();
			if ( PRINT_TIME_PER_STEP )
				System.out.println("Block matching 2 took: " + ((endTime - startTime) / 1000.0) + " sec");
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

		if ( param.useLocalSmoothnessFilter )
		{
			startTime = System.currentTimeMillis();
			System.out.println( layer1 + " < " + layer2 + ": found " + pm21.size() + " correspondence candidates." );
			localSmoothnessFilterModel.localSmoothnessFilter( pm21, pm21, localRegionSigma, maxLocalEpsilon, param.maxLocalTrust );
			System.out.println( layer1 + " < " + layer2 + ": " + pm21.size() + " candidates passed local smoothness filter." );
			endTime = System.currentTimeMillis();
			if ( PRINT_TIME_PER_STEP )
				System.out.println("local smooth filter 2 took: " + ((endTime - startTime) / 1000.0) + " sec");
		}
		else
		{
			System.out.println( layer1 + " < " + layer2 + ": found " + pm21.size() + " correspondences." );
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
		
/*
				for ( final PointMatch pm : pm21 )
				{
					final Vertex p1 = ( Vertex )pm.getP1();
					final Vertex p2 = new Vertex( pm.getP2() );
					p1.addSpring( p2, new Spring( 0, springConstant ) );
					m1.addPassiveVertex( p2 );
				}
*/				
		/*
		 * adding Tiles to the initialing TileConfiguration, adding a Tile
		 * multiple times does not harm because the TileConfiguration is
		 * backed by a Set. 
		 */
/*
				if ( pm21.size() > model.getMinNumMatches() )
				{
					initMeshes.addTile( t1 );
					initMeshes.addTile( t2 );
					t2.connect( t1, pm21 );
				}
*/
		// Remove Vertex (spring mesh) details from points
		final ArrayList< PointMatch > pm21_strip = new ArrayList< PointMatch >();
		for (PointMatch pm: pm21)
		{
			PointMatch actualPm;
			try {
				actualPm = new PointMatch(
						new Point( pm.getP1().getL(), scaleDownModel.applyInverse( pm.getP1().getW() ) ),
						new Point( pm.getP2().getL(), scaleDownModel.applyInverse( pm.getP2().getW() ) )
						);
				pm21_strip.add( actualPm );
			} catch (NoninvertibleModelException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		// TODO: Export / Import master sprint mesh vertices no calculated  individually per tile (v1, v2).
		corr_data.add(new CorrespondenceSpec(
				mipmapLevel,
				param.inputfile2,
				param.inputfile1,
				pm21_strip,
				( pm21.size() > model.getMinNumMatches() ) ));


		return corr_data;
	}

	private static List< CorrespondenceSpec > matchLayersByMaxPMCCByRectangle(
			final Params param,
			final AbstractModel< ? > model,
			final TileSpec[] ts1,
			final TileSpec[] ts2,
			int mipmapLevel,
			final int rowFromPixel,
			final int rowToPixel,
			final int colFromPixel,
			final int colToPixel )
	{
		final List< CorrespondenceSpec > corr_data = new ArrayList< CorrespondenceSpec >();

		System.out.println( "Finding matches in the following rectangle: [" + 
				colFromPixel + ", " + rowFromPixel + "] -> [" + colToPixel + ", " + rowToPixel + "]" );
/*
		final TileConfiguration initMeshes = new TileConfiguration();
*/
		final int layer1 = ts1[0].layer;
		final int layer2 = ts2[0].layer;
		

		// Compute bounding box of the two layers
		
		
		long startTime = System.currentTimeMillis();
		// Create the meshes for the tiles
		/*
		final List< TileSpec[] > tsList = new ArrayList< TileSpec[] >();
		tsList.add( ts1 );
		tsList.add( ts2 );
		
		final List< SpringMesh > meshes = Utils.createMeshes(
				tsList, param.imageWidth, param.imageHeight,
				param.springLengthSpringMesh, param.stiffnessSpringMesh,
				param.maxStretchSpringMesh, param.layerScale, param.dampSpringMesh );
		*/
		
		final int meshWidth = ( int )Math.ceil( param.imageWidth * param.layerScale );
		final int meshHeight = ( int )Math.ceil( param.imageHeight * param.layerScale );
		final SpringMesh[] meshes = new SpringMesh[2];
		for ( int i = 0; i < meshes.length; i++ )
			meshes[i] = new SpringMesh(
					param.resolutionSpringMesh,
					meshWidth,
					meshHeight,
					param.stiffnessSpringMesh,
					param.maxStretchSpringMesh * param.layerScale,
					param.dampSpringMesh );
		
		long endTime = System.currentTimeMillis();
		if ( PRINT_TIME_PER_STEP )
			System.out.println("Creating mesh took: " + ((endTime - startTime) / 1000.0) + " sec");

		//final int blockRadius = Math.max( 32, meshWidth / p.resolutionSpringMesh / 2 );
//		final int param_blockRadius = param.imageWidth / param.resolution / 2;
		final int orig_param_resolution = 16;
		final int param_blockRadius;
		if ( param.blockRadius < 0 )
			param_blockRadius = param.imageWidth / orig_param_resolution / 2;
		else
			param_blockRadius = param.blockRadius;
		final int blockRadius = Math.max( 16, mpicbg.util.Util.roundPos( param.layerScale * param_blockRadius ) );
//		final int blockRadius = Math.max( mpicbg.util.Util.roundPos( 16 / param.layerScale ), param.blockRadius );
		
		System.out.println( "effective block radius = " + blockRadius );
		
		/* scale pixel distances */
		final int searchRadius = ( int )Math.round( param.layerScale * param.searchRadius );
//		final int searchRadius = param.searchRadius;
		final float localRegionSigma = param.layerScale * param.localRegionSigma;
		final float maxLocalEpsilon = param.layerScale * param.maxLocalEpsilon;
//		final float localRegionSigma = param.localRegionSigma;
//		final float maxLocalEpsilon = param.maxLocalEpsilon;
		
		startTime = System.currentTimeMillis();
		final AbstractModel< ? > localSmoothnessFilterModel = Util.createModel( param.localModelIndex );
		endTime = System.currentTimeMillis();
		if ( PRINT_TIME_PER_STEP )
			System.out.println("Creating model took: " + ((endTime - startTime) / 1000.0) + " sec");

		
		final SpringMesh m1 = meshes[ 0 ];
		final SpringMesh m2 = meshes[ 1 ];

		final ArrayList< PointMatch > pm12 = new ArrayList< PointMatch >();
		final ArrayList< PointMatch > pm21 = new ArrayList< PointMatch >();

		final ArrayList< Vertex > v1 = m1.getVertices();
		final ArrayList< Vertex > v2 = m2.getVertices();

		// filter out vertices that are not in the rectangle
		final float fromX = colFromPixel * param.layerScale;
		final float toX = colToPixel * param.layerScale;
		final float fromY = rowFromPixel * param.layerScale;
		final float toY = rowToPixel * param.layerScale;
		
		final ArrayList< Vertex > v1Filtered = new ArrayList< Vertex >();
		for ( Vertex v : v1 )
		{
			float[] l = v.getL();
			if ( ( fromX <= l[0] ) && ( l[0] < toX ) &&
				 ( fromY <= l[1] ) && ( l[1] < toY ) )
				v1Filtered.add( v );
		}

		final ArrayList< Vertex > v2Filtered = new ArrayList< Vertex >();
		for ( Vertex v : v2 )
		{
			float[] l = v.getL();
			if ( ( fromX <= l[0] ) && ( l[0] < toX ) &&
				 ( fromY <= l[1] ) && ( l[1] < toY ) )
				v2Filtered.add( v );
		}

				
		/* Load images and masks into FloatProcessor objects */		
		System.out.println( "Rendering layers" );
		startTime = System.currentTimeMillis();
		final TileSpecsImage layer1Img = new TileSpecsImage( ts1 );
		final TileSpecsImage layer2Img = new TileSpecsImage( ts2 );
		
		layer1Img.setThreadsNum( param.numThreads );
		layer2Img.setThreadsNum( param.numThreads );
		
		/*
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
		*/

		// TODO - find rendering bounding box for each of the images
		// Get images size
		final BoundingBox layer1BBox = layer1Img.getBoundingBox();
		final BoundingBox layer2BBox = layer2Img.getBoundingBox();

		// Get rendering bounding box for image 1
		BoundingBox renderLayer1BBox = new BoundingBox( colFromPixel, colToPixel, rowFromPixel, rowToPixel );
		float[] img1OriginPoint = { colFromPixel, rowFromPixel };
		model.applyInPlace( img1OriginPoint );
		// Stretch the rendering bounding box of image 1 according to the transformation between the layers
		final BoundingBox renderLayer1BBoxTransformed = renderLayer1BBox.apply2DAffineTransformation( ( CoordinateTransform )model );
		renderLayer1BBox.extendByBoundingBox( renderLayer1BBoxTransformed );
		// Add a small buffer to the rendering bounding box of image 1 according to the search radius
		renderLayer1BBox.extendByMinMax(
				renderLayer1BBox.getStartPoint().getX() - param.searchRadius,
				renderLayer1BBox.getEndPoint().getX() + param.searchRadius + param.blockRadius,
				renderLayer1BBox.getStartPoint().getY() - param.searchRadius,
				renderLayer1BBox.getEndPoint().getY() + param.searchRadius + param.blockRadius );

		// Get rendering bounding box for image 2
		BoundingBox renderLayer2BBox = new BoundingBox( colFromPixel, colToPixel, rowFromPixel, rowToPixel );
		float[] img2OriginPoint = { colFromPixel, rowFromPixel };
		( ( InvertibleCoordinateTransform )model ).createInverse().applyInPlace( img2OriginPoint );
		// Stretch the rendering bounding box of image 1 according to the transformation between the layers
		final BoundingBox renderLayer2BBoxTransformed = renderLayer2BBox.apply2DAffineTransformation( ( ( InvertibleCoordinateTransform )model ).createInverse() );
		renderLayer2BBox.extendByBoundingBox( renderLayer2BBoxTransformed );
		// Add a small buffer to the rendering bounding box of image 2 according to the search radius
		renderLayer2BBox.extendByMinMax(
				renderLayer2BBox.getStartPoint().getX() - param.searchRadius,
				renderLayer2BBox.getEndPoint().getX() + param.searchRadius + param.blockRadius,
				renderLayer2BBox.getStartPoint().getY() - param.searchRadius,
				renderLayer2BBox.getEndPoint().getY() + param.searchRadius + param.blockRadius );
		
		// Translate the to be matches points according to the new rendered window
		for ( Vertex v : v1Filtered )
		{
			float[] l = v.getL();
			l[ 0 ] -= renderLayer1BBox.getStartPoint().getX() * param.layerScale;
			l[ 1 ] -= renderLayer1BBox.getStartPoint().getY() * param.layerScale;
		}
		for ( Vertex v : v2Filtered )
		{
			float[] l = v.getL();
			l[ 0 ] -= renderLayer2BBox.getStartPoint().getX() * param.layerScale;
			l[ 1 ] -= renderLayer2BBox.getStartPoint().getY() * param.layerScale;
		}
		
		// TODO: load the tile specs to FloatProcessor objects
//		final FloatProcessor ip1 = tilespecToFloatAndMask( layer1Img, layer1, mipmapLevel, param.layerScale, param.meshesDir1 );
//		final FloatProcessor ip2 = tilespecToFloatAndMask( layer2Img, layer2, mipmapLevel, param.layerScale, param.meshesDir2 );
//		final ByteProcessor ip1 = tilespecToByteAndMask( layer1Img, layer1, mipmapLevel, param.layerScale, param.meshesDir1 );
//		final ByteProcessor ip2 = tilespecToByteAndMask( layer2Img, layer2, mipmapLevel, param.layerScale, param.meshesDir2 );
		final FloatProcessor ip1 = tilespecToFloatAndMask( layer1Img, layer1, mipmapLevel, param.layerScale, param.meshesDir1, renderLayer1BBox );
		final FloatProcessor ip2 = tilespecToFloatAndMask( layer2Img, layer2, mipmapLevel, param.layerScale, param.meshesDir2, renderLayer2BBox );
		endTime = System.currentTimeMillis();
		if ( PRINT_TIME_PER_STEP )
			System.out.println("Creating images took: " + ((endTime - startTime) / 1000.0) + " sec");
		
		//final float springConstant  = 1.0f / ( layer2 - layer1 );
		

		// Scale the affine transformation
		final AffineTransform scaleDown = new AffineTransform();
		scaleDown.scale( param.layerScale, param.layerScale );
		final AffineModel2D scaleDownModel = new AffineModel2D();
		scaleDownModel.set( scaleDown );

		final TranslationModel2D tran1Model = new TranslationModel2D();
		tran1Model.set( renderLayer1BBox.getStartPoint().getX(), renderLayer1BBox.getStartPoint().getY() );

		final TranslationModel2D tran2Model = new TranslationModel2D();
		tran2Model.set( renderLayer2BBox.getStartPoint().getX(), renderLayer2BBox.getStartPoint().getY() );

		final TranslationModel2D tran12Model = new TranslationModel2D();
		tran12Model.set( renderLayer1BBox.getStartPoint().getX() - renderLayer2BBox.getStartPoint().getX(), renderLayer1BBox.getStartPoint().getY() - renderLayer2BBox.getStartPoint().getY() );

		final TranslationModel2D tran12ModelA = new TranslationModel2D();
		tran12ModelA.set( img1OriginPoint[0] - colFromPixel, img1OriginPoint[1] - rowFromPixel );
		final TranslationModel2D tran12ModelB = new TranslationModel2D();
		tran12ModelB.set( img2OriginPoint[0] - colFromPixel, img2OriginPoint[1] - rowFromPixel );
		
		final CoordinateTransformList< CoordinateTransform > scaledModel = new CoordinateTransformList< CoordinateTransform >();
		scaledModel.add( scaleDownModel.createInverse() ); // scale up
		scaledModel.add( tran1Model ); // translate the cropped img1 to the full img1 position
		scaledModel.add( ( CoordinateTransform )model ); // apply model (full img1 -> full img2)
		scaledModel.add( ( ( InvertibleCoordinateTransform )tran1Model ).createInverse() ); // translate back
		scaledModel.add( ( ( InvertibleCoordinateTransform )tran12Model ).createInverse() );
		//scaledModel.add( tran12ModelB );
		scaledModel.add( scaleDownModel ); // scale down
		
		final CoordinateTransformList< CoordinateTransform > scaledInverseModel = new CoordinateTransformList< CoordinateTransform >();
		scaledInverseModel.add( scaleDownModel.createInverse() ); // scale up
		scaledInverseModel.add( tran2Model ); // translate the cropped img2 to the full img2 position 
		scaledInverseModel.add( ( ( InvertibleCoordinateTransform )model ).createInverse() ); // apply inverse model (full img2 -> full img1)
		scaledInverseModel.add( ( ( InvertibleCoordinateTransform )tran2Model ).createInverse() ); // translate back
		scaledInverseModel.add( tran12Model );
		//scaledInverseModel.add( tran12ModelA );
		scaledInverseModel.add( scaleDownModel ); // scale down

		
		try
		{
			startTime = System.currentTimeMillis();
			BlockMatching.matchByMaximalPMCC(
					ip1,
					ip2,
					null,//ip1Mask,
					null,//ip2Mask,
					1.0f,
					scaledInverseModel, //( ( InvertibleCoordinateTransform )model ).createInverse(),
					blockRadius,
					blockRadius,
					searchRadius,
					searchRadius,
					param.minR,
					param.rodR,
					param.maxCurvatureR,
					v1Filtered,
					pm12,
					new ErrorStatistic( 1 ) );
			endTime = System.currentTimeMillis();
			if ( PRINT_TIME_PER_STEP )
				System.out.println("Block matching 1 took: " + ((endTime - startTime) / 1000.0) + " sec");
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

//		// Move the match results back to their "original" coordinate system
//		TranslationModel2D tran_back12 = new TranslationModel2D();
//		tran_back12.set( renderLayer1BBox.getStartPoint().getX() * param.layerScale, renderLayer1BBox.getStartPoint().getY() * param.layerScale );
//		for ( PointMatch pm: pm12 )
//		{
//			
//			pm.getP1().getL();
//			pm.apply( tran_back12 );
//		}

		
		if ( param.useLocalSmoothnessFilter )
		{

			startTime = System.currentTimeMillis();
			System.out.println( layer1 + " > " + layer2 + ": found " + pm12.size() + " correspondence candidates." );
			localSmoothnessFilterModel.localSmoothnessFilter( pm12, pm12, localRegionSigma, maxLocalEpsilon, param.maxLocalTrust );
			System.out.println( layer1 + " > " + layer2 + ": " + pm12.size() + " candidates passed local smoothness filter." );
			endTime = System.currentTimeMillis();
			if ( PRINT_TIME_PER_STEP )
				System.out.println("local smooth filter 1 took: " + ((endTime - startTime) / 1000.0) + " sec");
		}
		else
		{
			System.out.println( layer1 + " > " + layer2 + ": found " + pm12.size() + " correspondences." );
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
/*				
				for ( final PointMatch pm : pm12 )
				{
					final Vertex p1 = ( Vertex )pm.getP1();
					final Vertex p2 = new Vertex( pm.getP2() );
					p1.addSpring( p2, new Spring( 0, springConstant ) );
					m2.addPassiveVertex( p2 );
				}
*/

		/*
		 * adding Tiles to the initialing TileConfiguration, adding a Tile
		 * multiple times does not harm because the TileConfiguration is
		 * backed by a Set. 
		 */
/*
				if ( pm12.size() > model.getMinNumMatches() )
				{
					initMeshes.addTile( t1 );
					initMeshes.addTile( t2 );
					t1.connect( t2, pm12 );
				}
*/
		
		// Remove Vertex (spring mesh) details from points
		final ArrayList< PointMatch > pm12_strip = new ArrayList< PointMatch >();
		for (PointMatch pm: pm12)
		{
			PointMatch actualPm;
//			try {
				// translate the pointmatch local coordinates to the actual non-rectangular coordinate system
				float[] pm1L = pm.getP1().getL();
				pm1L[0] += renderLayer1BBox.getStartPoint().getX() * param.layerScale;
				pm1L[1] += renderLayer1BBox.getStartPoint().getY() * param.layerScale;
				float[] pm2L = pm.getP2().getL();
				pm2L[0] += renderLayer2BBox.getStartPoint().getX() * param.layerScale;
				pm2L[1] += renderLayer2BBox.getStartPoint().getY() * param.layerScale;
				// translate the pointmatch world coordinates to the actual non-rectangular coordinate system
				float[] pm1W = pm.getP1().getW();
				pm1W[0] = ( pm1W[0] / param.layerScale ) + renderLayer1BBox.getStartPoint().getX();
				pm1W[1] = ( pm1W[1] / param.layerScale ) + renderLayer1BBox.getStartPoint().getY();
				float[] pm2W = pm.getP2().getW();
				pm2W[0] = ( pm2W[0] / param.layerScale ) + renderLayer2BBox.getStartPoint().getX();
				pm2W[1] = ( pm2W[1] / param.layerScale ) + renderLayer2BBox.getStartPoint().getY();
				actualPm = new PointMatch(
						new Point( pm1L, pm1W ),
						new Point( pm2L, pm2W )
						);
/*
				actualPm = new PointMatch(
						new Point( pm.getP1().getL(), scaleDownModel.applyInverse( pm.getP1().getW() ) ),
						new Point( pm.getP2().getL(), scaleDownModel.applyInverse( pm.getP2().getW() ) )
						);
*/
				pm12_strip.add( actualPm );
//			} catch (NoninvertibleModelException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
		}
		
		// TODO: Export / Import master sprint mesh vertices no calculated  individually per tile (v1, v2).
		corr_data.add(new CorrespondenceSpec(
				mipmapLevel,
				param.inputfile1,
				param.inputfile2,
				pm12_strip,
				( pm12.size() > model.getMinNumMatches() ) ));

		try
		{
			startTime = System.currentTimeMillis();
			BlockMatching.matchByMaximalPMCC(
					ip2,
					ip1,
					null,//ip2Mask,
					null,//ip1Mask,
					1.0f,
					scaledModel, //model,
					blockRadius,
					blockRadius,
					searchRadius,
					searchRadius,
					param.minR,
					param.rodR,
					param.maxCurvatureR,
					v2Filtered,
					pm21,
					new ErrorStatistic( 1 ) );
			endTime = System.currentTimeMillis();
			if ( PRINT_TIME_PER_STEP )
				System.out.println("Block matching 2 took: " + ((endTime - startTime) / 1000.0) + " sec");
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

//		// Move the match results back to their "original" coordinate system
//		TranslationModel2D tran_back21 = new TranslationModel2D();
//		tran_back21.set( renderLayer2BBox.getStartPoint().getX() * param.layerScale, renderLayer2BBox.getStartPoint().getY() * param.layerScale );
//		for ( PointMatch pm: pm21 )
//		{
//			pm.apply( tran_back21 );
//		}

		if ( param.useLocalSmoothnessFilter )
		{
			startTime = System.currentTimeMillis();
			System.out.println( layer1 + " < " + layer2 + ": found " + pm21.size() + " correspondence candidates." );
			localSmoothnessFilterModel.localSmoothnessFilter( pm21, pm21, localRegionSigma, maxLocalEpsilon, param.maxLocalTrust );
			System.out.println( layer1 + " < " + layer2 + ": " + pm21.size() + " candidates passed local smoothness filter." );
			endTime = System.currentTimeMillis();
			if ( PRINT_TIME_PER_STEP )
				System.out.println("local smooth filter 2 took: " + ((endTime - startTime) / 1000.0) + " sec");
		}
		else
		{
			System.out.println( layer1 + " < " + layer2 + ": found " + pm21.size() + " correspondences." );
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
		
/*
				for ( final PointMatch pm : pm21 )
				{
					final Vertex p1 = ( Vertex )pm.getP1();
					final Vertex p2 = new Vertex( pm.getP2() );
					p1.addSpring( p2, new Spring( 0, springConstant ) );
					m1.addPassiveVertex( p2 );
				}
*/				
		/*
		 * adding Tiles to the initialing TileConfiguration, adding a Tile
		 * multiple times does not harm because the TileConfiguration is
		 * backed by a Set. 
		 */
/*
				if ( pm21.size() > model.getMinNumMatches() )
				{
					initMeshes.addTile( t1 );
					initMeshes.addTile( t2 );
					t2.connect( t1, pm21 );
				}
*/
		// Remove Vertex (spring mesh) details from points
		final ArrayList< PointMatch > pm21_strip = new ArrayList< PointMatch >();
		for (PointMatch pm: pm21)
		{
			PointMatch actualPm;
			// translate the pointmatch local coordinates to the actual non-rectangular coordinate system
			float[] pm1L = pm.getP1().getL();
			pm1L[0] += renderLayer2BBox.getStartPoint().getX() * param.layerScale;
			pm1L[1] += renderLayer2BBox.getStartPoint().getY() * param.layerScale;
			float[] pm2L = pm.getP2().getL();
			pm2L[0] += renderLayer1BBox.getStartPoint().getX() * param.layerScale;
			pm2L[1] += renderLayer1BBox.getStartPoint().getY() * param.layerScale;
			// translate the pointmatch world coordinates to the actual non-rectangular coordinate system
			float[] pm1W = pm.getP1().getW();
			pm1W[0] = ( pm1W[0] / param.layerScale ) + renderLayer2BBox.getStartPoint().getX();
			pm1W[1] = ( pm1W[1] / param.layerScale ) + renderLayer2BBox.getStartPoint().getY();
			float[] pm2W = pm.getP2().getW();
			pm2W[0] = ( pm2W[0] / param.layerScale ) + renderLayer1BBox.getStartPoint().getX();
			pm2W[1] = ( pm2W[1] / param.layerScale ) + renderLayer1BBox.getStartPoint().getY();
			actualPm = new PointMatch(
					new Point( pm1L, pm1W ),
					new Point( pm2L, pm2W )
					);
			pm21_strip.add( actualPm );

			/*
			try {
				actualPm = new PointMatch(
						new Point( pm.getP1().getL(), scaleDownModel.applyInverse( pm.getP1().getW() ) ),
						new Point( pm.getP2().getL(), scaleDownModel.applyInverse( pm.getP2().getW() ) )
						);
				pm21_strip.add( actualPm );
			} catch (NoninvertibleModelException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			*/
		}

		// TODO: Export / Import master sprint mesh vertices no calculated  individually per tile (v1, v2).
		corr_data.add(new CorrespondenceSpec(
				mipmapLevel,
				param.inputfile2,
				param.inputfile1,
				pm21_strip,
				( pm21.size() > model.getMinNumMatches() ) ));


		return corr_data;
	}

	private static List<CorrespondenceSpec> createEmptyCorrespondece(
			final int layer1,
			final int layer2,
			final Params param,
			final int mipmapLevel )
	{
		final List< CorrespondenceSpec > corr_data = new ArrayList< CorrespondenceSpec >();

		corr_data.add(new CorrespondenceSpec(
				mipmapLevel,
				param.inputfile1,
				param.inputfile2,
				null,
				false ));

		corr_data.add(new CorrespondenceSpec(
				mipmapLevel,
				param.inputfile2,
				param.inputfile1,
				null,
				false ));

		return corr_data;
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

		/* Open the tilespecs */
		TileSpec[] tilespecs1 = TileSpecUtils.readTileSpecFile( params.inputfile1 );
		TileSpec[] tilespecs2 = TileSpecUtils.readTileSpecFile( params.inputfile2 );
		
		/* open the models */
		long startTime = System.currentTimeMillis();
		CoordinateTransform model = null;
		try
		{
			final ModelSpec[] modelSpecs;
			final Gson gson = new Gson();
			URL url = new URL( params.modelsfile1 );
			modelSpecs = gson.fromJson( new InputStreamReader( url.openStream() ), ModelSpec[].class );
			
			for ( ModelSpec ms : modelSpecs )
			{
				if ( (( params.inputfile1.equals( ms.url1 ) ) && ( params.inputfile2.equals( ms.url2 ) )) ||
						(( params.inputfile1.equals( ms.url2 ) ) && ( params.inputfile2.equals( ms.url1 ) )) )
				{
					model = ms.createModel();
				}
			}

			if ( model == null )
			{
				if ( params.autoAddModel )
					model = new IdentityModel();
				else
				{
					// Write an "empty" file
					final List< CorrespondenceSpec > corr_data = createEmptyCorrespondece(
							tilespecs1[0].layer, tilespecs2[0].layer,
							params, mipmapLevel );
					System.out.println( "Writing an empty correspondece match file between layers: " + params.inputfile1 + " and " + params.inputfile2 );
					try {
						Writer writer = new FileWriter(params.targetPath);
				        //Gson gson = new GsonBuilder().create();
				        Gson gsonOut = new GsonBuilder().setPrettyPrinting().create();
				        gsonOut.toJson(corr_data, writer);
				        writer.close();
				        return;
				    }
					catch ( final IOException e )
					{
						System.err.println( "Error writing JSON file: " + params.targetPath );
						e.printStackTrace( System.err );
					}
				}
			}
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
		
		long endTime = System.currentTimeMillis();
		System.out.println("Parsing files took: " + ((endTime - startTime) / 1000.0) + " ms");
		
		final List< CorrespondenceSpec > corr_data;
		if ( params.rectsPerDim == 1 ) // A single rectangle for the entire image
		{
			startTime = System.currentTimeMillis();
			corr_data = matchLayersByMaxPMCC( 
					params, (AbstractModel< ? >)model,
					tilespecs1, tilespecs2, mipmapLevel );
			endTime = System.currentTimeMillis();
			System.out.println("Entire match process took: " + ((endTime - startTime) / 1000.0) + " ms");
		}
		else
		{
			startTime = System.currentTimeMillis();
			//final int meshWidth = ( int )Math.ceil( params.imageWidth * params.layerScale );
			//final int meshHeight = ( int )Math.ceil( params.imageHeight * params.layerScale );

			final int rectWidth = params.imageWidth / params.rectsPerDim;
			final int rectHeight = params.imageHeight / params.rectsPerDim;
			final List< List< PointMatch > > matches_layer_1to2 = new ArrayList< List< PointMatch > >();
			final List< List< PointMatch > > matches_layer_2to1 = new ArrayList< List< PointMatch > >();

			int matchesNum_1to2 = 0;
			int matchesNum_2to1 = 0;

			for ( int row = 0; row < params.rectsPerDim; row++ )
			{
				int fromRowPixel = row * rectHeight;
				int toRowPixel;
				if ( row == params.rectsPerDim - 1 )
					toRowPixel = params.imageHeight;
				else
					toRowPixel = fromRowPixel + rectHeight;
				
				for ( int col = 0; col < params.rectsPerDim; col++ )
				{
					int fromColPixel = col * rectWidth;
					int toColPixel;
					if ( col == params.rectsPerDim - 1 )
						toColPixel = params.imageWidth;
					else
						toColPixel = fromColPixel + rectWidth;
					
					CorrespondenceSpec[] rectData = loadIntermediateResults( params, row, col );
					if ( rectData == null )
					{
						
						final List< CorrespondenceSpec > rectDataList = matchLayersByMaxPMCCByRectangle( 
								params, (AbstractModel< ? >)model,
								tilespecs1, tilespecs2, mipmapLevel,
								fromRowPixel, toRowPixel, fromColPixel, toColPixel );
						/*
						final List< CorrespondenceSpec > rectDataList = matchLayersByMaxPMCCByRectangle( 
								params, (AbstractModel< ? >)model,
								tilespecs1, tilespecs2, mipmapLevel,
								0, params.imageHeight, 0, params.imageWidth );
						*/
						
						// Save the intermediate data
						saveIntermediateResults( rectDataList, params, row, col );
						
						matches_layer_1to2.add( rectDataList.get( 0 ).correspondencePointPairs );
						matches_layer_2to1.add( rectDataList.get( 1 ).correspondencePointPairs );
						
						matchesNum_1to2 += rectDataList.get( 0 ).correspondencePointPairs.size();
						matchesNum_2to1 += rectDataList.get( 1 ).correspondencePointPairs.size();
						
					}
					else
					{
						matches_layer_1to2.add( rectData[ 0 ].correspondencePointPairs );
						matches_layer_2to1.add( rectData[ 1 ].correspondencePointPairs );

						matchesNum_1to2 += rectData[ 0 ].correspondencePointPairs.size();
						matchesNum_2to1 += rectData[ 1 ].correspondencePointPairs.size();
					}
				}
			}
			endTime = System.currentTimeMillis();
			System.out.println("Entire match process by rectangles took: " + ((endTime - startTime) / 1000.0) + " ms");

			List< PointMatch > pm12 = new ArrayList< PointMatch >( matchesNum_1to2 );
			List< PointMatch > pm21 = new ArrayList< PointMatch >( matchesNum_2to1 );
			for ( int rect = 0; rect < matches_layer_1to2.size(); rect++ )
			{
				pm12.addAll( matches_layer_1to2.get( rect ) );
				pm21.addAll( matches_layer_2to1.get( rect ) );
			}

			corr_data = new ArrayList<CorrespondenceSpec>();

			corr_data.add(new CorrespondenceSpec(
					mipmapLevel,
					params.inputfile1,
					params.inputfile2,
					pm12,
					( pm12.size() > ((AbstractModel< ? >)model).getMinNumMatches() ) ));

			corr_data.add(new CorrespondenceSpec(
					mipmapLevel,
					params.inputfile2,
					params.inputfile1,
					pm21,
					( pm21.size() > ((AbstractModel< ? >)model).getMinNumMatches() ) ));

			
		}

		// In case no correspondence points are found, write an "empty" file
		startTime = System.currentTimeMillis();
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
		endTime = System.currentTimeMillis();
		System.out.println("Writing output took: " + ((endTime - startTime) / 1000.0) + " ms");

	}

}

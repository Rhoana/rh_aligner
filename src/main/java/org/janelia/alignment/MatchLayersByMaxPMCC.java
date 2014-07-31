package org.janelia.alignment;

import ij.process.ColorProcessor;
import ij.process.FloatProcessor;

import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

import mpicbg.ij.blockmatching.BlockMatching;
import mpicbg.models.AbstractModel;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.ErrorStatistic;
import mpicbg.models.IdentityModel;
import mpicbg.models.InvertibleCoordinateTransform;
import mpicbg.models.PointMatch;
import mpicbg.models.SpringMesh;
import mpicbg.models.Vertex;
import mpicbg.trakem2.align.Util;

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

        @Parameter( names = "--autoAddModel", description = "Automatically add the Identity model in case a model is not found", required = false )
        private boolean autoAddModel = false;

        @Parameter( names = "--fixedLayers", description = "Fixed layer numbers (space separated)", variableArity = true, required = false )
        public List<Integer> fixedLayers = new ArrayList<Integer>();
        
        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        public float layerScale = 0.1f;
        
        @Parameter( names = "--searchRadius", description = "Search window radius", required = false )
        public int searchRadius = 200;
        
        @Parameter( names = "--blockRadius", description = "Matching block radius", required = false )
        public int blockRadius = 579;
                
        @Parameter( names = "--resolution", description = "Resolution", required = false )
        public int resolution = 16;
        
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
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
	}
	
	private MatchLayersByMaxPMCC() {}
	
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
			final float layerScale )
	{
		final ColorProcessor cp = layerTileSpecsImage.render( layer, mipmapLevel, layerScale );
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
			int mipmapLevel,
			final List< Integer > fixedLayers )
	{
		final List< CorrespondenceSpec > corr_data = new ArrayList< CorrespondenceSpec >();

/*
		final TileConfiguration initMeshes = new TileConfiguration();
*/
		final int layer1 = ts1[0].layer;
		final int layer2 = ts2[0].layer;
		
		final boolean layer1Fixed = fixedLayers.contains( layer1 );
		final boolean layer2Fixed = fixedLayers.contains( layer2 );

		if ( layer1Fixed && layer2Fixed )
		{
			// Both layers are fixed, nothing to do...
			// Returns an empty correspondence spec list
			return corr_data;
		}

		// Compute bounding box of the two layers
		
		
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

		//final int blockRadius = Math.max( 32, meshWidth / p.resolutionSpringMesh / 2 );
		final int param_blockRadius = param.imageWidth / param.resolution / 2;
		final int blockRadius = Math.max( 16, mpicbg.util.Util.roundPos( param.layerScale * param_blockRadius ) );
		
		System.out.println( "effective block radius = " + blockRadius );
		
		/* scale pixel distances */
		final int searchRadius = ( int )Math.round( param.layerScale * param.searchRadius );
		final float localRegionSigma = param.layerScale * param.localRegionSigma;
		final float maxLocalEpsilon = param.layerScale * param.maxLocalEpsilon;
		
		final AbstractModel< ? > localSmoothnessFilterModel = Util.createModel( param.localModelIndex );

		
		final SpringMesh m1 = meshes[0];
		final SpringMesh m2 = meshes[1];

		final ArrayList< PointMatch > pm12 = new ArrayList< PointMatch >();
		final ArrayList< PointMatch > pm21 = new ArrayList< PointMatch >();

		final ArrayList< Vertex > v1 = m1.getVertices();
		final ArrayList< Vertex > v2 = m2.getVertices();
		
				
		//if ( !( layer1Fixed && layer2Fixed ) )
		//{
			/* Load images and masks into FloatProcessor objects */		
			final TileSpecsImage layer1Img = new TileSpecsImage( ts1 );
			final TileSpecsImage layer2Img = new TileSpecsImage( ts2 );
			
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
			tilespecToFloatAndMask( layer1Img, layer1, ip1, ip1Mask, mipmapLevel, param.layerScale );
			tilespecToFloatAndMask( layer2Img, layer2, ip2, ip2Mask, mipmapLevel, param.layerScale );
			
			//final float springConstant  = 1.0f / ( layer2 - layer1 );
			
			if ( ! layer1Fixed )
			{
				try
				{
					BlockMatching.matchByMaximalPMCC(
							ip1,
							ip2,
							null,//ip1Mask,
							null,//ip2Mask,
							1.0f,
							( ( InvertibleCoordinateTransform )model ).createInverse(),
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
					System.out.println( layer1 + " > " + layer2 + ": found " + pm12.size() + " correspondence candidates." );
					localSmoothnessFilterModel.localSmoothnessFilter( pm12, pm12, localRegionSigma, maxLocalEpsilon, param.maxLocalTrust );
					System.out.println( layer1 + " > " + layer2 + ": " + pm12.size() + " candidates passed local smoothness filter." );
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
				// TODO: Export / Import master sprint mesh vertices no calculated  individually per tile (v1, v2).
				corr_data.add(new CorrespondenceSpec(
						mipmapLevel,
						param.inputfile1,
						param.inputfile2,
						pm12,
						( pm12.size() > model.getMinNumMatches() ) ));
				
			}

			if ( !layer2Fixed )
			{
				try
				{
					BlockMatching.matchByMaximalPMCC(
							ip2,
							ip1,
							null,//ip2Mask,
							null,//ip1Mask,
							1.0f,
							model,
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
					System.out.println( layer1 + " < " + layer2 + ": found " + pm21.size() + " correspondence candidates." );
					localSmoothnessFilterModel.localSmoothnessFilter( pm21, pm21, localRegionSigma, maxLocalEpsilon, param.maxLocalTrust );
					System.out.println( layer1 + " < " + layer2 + ": " + pm21.size() + " candidates passed local smoothness filter." );
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
				// TODO: Export / Import master sprint mesh vertices no calculated  individually per tile (v1, v2).
				corr_data.add(new CorrespondenceSpec(
						mipmapLevel,
						param.inputfile2,
						param.inputfile1,
						pm21,
						( pm21.size() > model.getMinNumMatches() ) ));


			}
			
			//System.out.println( layer1 + " <> " + layer2 + " spring constant = " + springConstant );
		//}

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
		
		/* open the models */
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
					throw new RuntimeException( "Error: model between the given two tilespecs was not found. ");
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
		
		// The mipmap level to work on
		// TODO: Should be a parameter from the user,
		//       and decide whether or not to create the mipmaps if they are missing
		int mipmapLevel = 0;

		TileSpec[] tilespecs1 = TileSpecUtils.readTileSpecFile( params.inputfile1 );
		TileSpec[] tilespecs2 = TileSpecUtils.readTileSpecFile( params.inputfile2 );
		
		final List< CorrespondenceSpec > corr_data = matchLayersByMaxPMCC( 
				params, (AbstractModel< ? >)model,
				tilespecs1, tilespecs2, mipmapLevel,
				params.fixedLayers );
		

		// In case no correspondence points are found, write an "empty" file
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

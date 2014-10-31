package org.janelia.alignment;

import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.Roi;
import ij.process.Blitter;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;

import java.awt.Color;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutionException;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RealPoint;
import net.imglib2.collection.KDTree;
import net.imglib2.collection.RealPointSampleList;
import net.imglib2.exception.ImgLibException;
import net.imglib2.img.imageplus.ImagePlusImg;
import net.imglib2.img.imageplus.ImagePlusImgFactory;
import net.imglib2.neighborsearch.NearestNeighborSearch;
import net.imglib2.neighborsearch.NearestNeighborSearchOnKDTree;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.ARGBType;
import mpicbg.ij.blockmatching.BlockMatching;
import mpicbg.models.ErrorStatistic;
import mpicbg.models.IllDefinedDataPointsException;
import mpicbg.models.Model;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.PointMatch;
import mpicbg.models.SpringMesh;
import mpicbg.models.TranslationModel2D;
import mpicbg.models.Vertex;
import mpicbg.util.Timer;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

/**
 * Takes two sections and executes the block matching (MatchLayersByMaxPMCC)
 * process on them.
 */
public class TestBlockMatchingParameters {
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--tilespecFiles", description = "Tilespec json files  (space separated) or a single file containing a line-separated list of json files", variableArity = true, required = true )
        public List<String> tileSpecFiles = new ArrayList<String>();

        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        public float layerScale = 1.0f;
        
        @Parameter( names = "--searchRadius", description = "Search window radius", required = false )
        public int searchRadius = 50;
        
        @Parameter( names = "--blockRadius", description = "Matching block radius", required = false )
        public int blockRadius = 50;
                
//        @Parameter( names = "--resolution", description = "Resolution", required = false )
//        public int resolution = 16;
        
        @Parameter( names = "--minR", description = "minR", required = false )
        public float minR = 0.1f;
        
        @Parameter( names = "--maxCurvatureR", description = "maxCurvatureR", required = false )
        public float maxCurvatureR = 1000.0f;
        
        @Parameter( names = "--rodR", description = "rodR", required = false )
        public float rodR = 1.0f;
        
        @Parameter( names = "--useLocalSmoothnessFilter", description = "useLocalSmoothnessFilter", required = false )
        public boolean useLocalSmoothnessFilter = false;
        
        @Parameter( names = "--localModelIndex", description = "localModelIndex", required = false )
        public int localModelIndex = 1;
        // 0 = "Translation", 1 = "Rigid", 2 = "Similarity", 3 = "Affine"
        
        @Parameter( names = "--localRegionSigma", description = "localRegionSigma", required = false )
        public float localRegionSigma = 65.0f;
        
        @Parameter( names = "--maxLocalEpsilon", description = "maxLocalEpsilon", required = false )
        public float maxLocalEpsilon = 12.0f;
        
        @Parameter( names = "--maxLocalTrust", description = "maxLocalTrust", required = false )
        public int maxLocalTrust = 3;
        
        //@Parameter( names = "--maxNumNeighbors", description = "maxNumNeighbors", required = false )
        //public float maxNumNeighbors = 3f;
        		
        @Parameter( names = "--resolutionSpringMesh", description = "resolutionSpringMesh", required = false )
        private int resolutionSpringMesh = 24;
        
        // Although not used, this needs to be here to comply with the MatchLayersByMaxPMCC parameters
        @Parameter( names = "--stiffnessSpringMesh", description = "stiffnessSpringMesh", required = false )
        public float stiffnessSpringMesh = 0.1f;
		
        // Although not used, this needs to be here to comply with the MatchLayersByMaxPMCC parameters
        @Parameter( names = "--dampSpringMesh", description = "dampSpringMesh", required = false )
        public float dampSpringMesh = 0.9f;
		
        // Although not used, this needs to be here to comply with the MatchLayersByMaxPMCC parameters
        @Parameter( names = "--maxStretchSpringMesh", description = "maxStretchSpringMesh", required = false )
        public float maxStretchSpringMesh = 2000.0f;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
	}
	
	private TestBlockMatchingParameters() { }

    final static int minGridSize = 8;
    final static boolean exportDisplacementVectors = true;

    /**
     * Generate an integer encoded 24bit RGB color that encodes a 2d vector
     * with amplitude being intensity and color being orientation.
     *
     * Only amplitudes in [0,1] will render into useful colors, so the vector
     * should be normalized to an expected max amplitude.
     * @param xs
     * @param ys
     * @return
     */
    final static private int colorVector( final float xs, final float ys )
    {
    	final double a = Math.sqrt( xs * xs + ys * ys );
    	if ( a == 0.0 ) return 0;

    	double o = ( Math.atan2( xs / a, ys / a ) + Math.PI ) / Math.PI * 3;

    	final double r, g, b;

    	if ( o < 3 )
    		r = Math.min( 1.0, Math.max( 0.0, 2.0 - o ) ) * a;
    	else
    		r = Math.min( 1.0, Math.max( 0.0, o - 4.0 ) ) * a;

    	o += 2;
    	if ( o >= 6 ) o -= 6;

    	if ( o < 3 )
    		g = Math.min( 1.0, Math.max( 0.0, 2.0 - o ) ) * a;
    	else
    		g = Math.min( 1.0, Math.max( 0.0, o - 4.0 ) ) * a;

    	o += 2;
    	if ( o >= 6 ) o -= 6;

    	if ( o < 3 )
    		b = Math.min( 1.0, Math.max( 0.0, 2.0 - o ) ) * a;
    	else
    		b = Math.min( 1.0, Math.max( 0.0, o - 4.0 ) ) * a;

    	return ( ( ( ( int )( r * 255 ) << 8 ) | ( int )( g * 255 ) ) << 8 ) | ( int )( b * 255 );
    }                
	
	private static ArrayList< PointMatch > match(
			final Params param,
			final int imageWidth,
			final int imageHeight,
			final FloatProcessor ip1,
			final FloatProcessor ip2,
			final FloatProcessor ip1Mask,
			final FloatProcessor ip2Mask )
			{
		final SpringMesh mesh = new SpringMesh( param.resolutionSpringMesh, imageWidth, imageHeight, 1, 1000, 0.9f );

		final ArrayList< PointMatch > pm12 = new ArrayList< PointMatch >();

		final Collection< Vertex > v1 = mesh.getVertices();

		final TranslationModel2D ct = new TranslationModel2D();

		try
		{
			BlockMatching.matchByMaximalPMCC(
					ip1,
					ip2,
					ip1Mask,
					ip2Mask,
					param.layerScale,
					ct,
					param.blockRadius,
					param.blockRadius,
					param.searchRadius,
					param.searchRadius,
					param.minR,
					param.rodR,
					param.maxCurvatureR,
					v1,
					pm12,
					new ErrorStatistic( 1 ) );
		}
		catch ( final InterruptedException e )
		{
			IJ.log( "Block Matching interrupted." );
			return pm12;
		}
		catch ( final ExecutionException e )
		{
			IJ.log( "Execution Exception occured during Block Matching." );
			e.printStackTrace();
			return pm12;
		}

		return pm12;
	}


	private static void filter( 
			final Params param,
			final ArrayList< PointMatch > pm12 )
	{
		final Model< ? > model = mpicbg.trakem2.align.Util.createModel( param.localModelIndex );
		try
		{
			model.localSmoothnessFilter( pm12, pm12, param.localRegionSigma, param.maxLocalEpsilon, param.maxLocalTrust );
		}
		catch ( final NotEnoughDataPointsException e )
		{
			pm12.clear();
		}
		catch ( final IllDefinedDataPointsException e )
		{
			pm12.clear();
		}
	}


	final static private RealPointSampleList< ARGBType > matches2ColorSamples( 
			final Params param,
			final Iterable< PointMatch > matches )
	{
		final RealPointSampleList< ARGBType > samples = new RealPointSampleList< ARGBType >( 2 );
		for ( final PointMatch match : matches )
		{
			final float[] p = match.getP1().getL();
			final float[] q = match.getP2().getW();
			final float dx = ( q[ 0 ] - p[ 0 ] ) / param.searchRadius;
			final float dy = ( q[ 1 ] - p[ 1 ] ) / param.searchRadius;

			final int rgb = colorVector( dx, dy );

			samples.add( new RealPoint( p ), new ARGBType( rgb ) );
		}
		return samples;
	}

	/*
	final private static RealPointSampleList< ARGBType > matches2ColorSamples2( 
			final Params param,
			final Iterable< PointMatch > matches )
	{
		final RealPointSampleList< ARGBType > samples = new RealPointSampleList< ARGBType >( 2 );
		for ( final PointMatch match : matches )
		{
			final float[] p = match.getP1().getL();
			final float[] q = match.getP2().getW();
			final float dx = ( q[ 0 ] - p[ 0 ] ) / param.searchRadius;
			final float dy = ( q[ 1 ] - p[ 1 ] ) / param.searchRadius;

			final int rgb = colorVector( dx, dy );

			samples.add( new RealPoint( q ), new ARGBType( rgb ) );
		}
		return samples;
	}
	*/

	final private static < T extends Type< T > > long drawNearestNeighbor(
			final IterableInterval< T > target,
			final NearestNeighborSearch< T > nnSearchSamples,
			final NearestNeighborSearch< T > nnSearchMask )
	{
		final Timer timer = new Timer();
		timer.start();
		final Cursor< T > c = target.localizingCursor();
		while ( c.hasNext() )
		{
			c.fwd();
			nnSearchSamples.search( c );
			nnSearchMask.search( c );
			if ( nnSearchSamples.getSquareDistance() <= nnSearchMask.getSquareDistance() )
				c.get().set( nnSearchSamples.getSampler().get() );
			else
				c.get().set( nnSearchMask.getSampler().get() );
		}
		return timer.stop();
	}
	
	private static void display( 
			final Params param,
			final int imageWidth,
			final int imageHeight,
			final ArrayList< PointMatch > pm12,
			final RealPointSampleList< ARGBType > maskSamples,
			final ImagePlus impTable,
			final ColorProcessor ipTable,
			final int w,
			final int h,
			final int i,
			final int j )
	{

		if ( pm12.size() > 0 )
		{
			final ImagePlusImgFactory< ARGBType > factory = new ImagePlusImgFactory< ARGBType >();

			final KDTree< ARGBType > kdtreeMatches = new KDTree< ARGBType >( matches2ColorSamples( param, pm12 ) );
			final KDTree< ARGBType > kdtreeMask = new KDTree< ARGBType >( maskSamples );

			/* nearest neighbor */
			final ImagePlusImg< ARGBType, ? > img = factory.create( new long[]{ imageWidth, imageHeight }, new ARGBType() );
			drawNearestNeighbor(
					img,
					new NearestNeighborSearchOnKDTree< ARGBType >( kdtreeMatches ),
					new NearestNeighborSearchOnKDTree< ARGBType >( kdtreeMask ) );

			final ImagePlus impVis;
			ColorProcessor ipVis;
			try
			{
				impVis = img.getImagePlus();
				ipVis = ( ColorProcessor )impVis.getProcessor();
				while ( ipVis.getWidth() > param.resolutionSpringMesh * minGridSize )
					ipVis = Downsampler.downsampleColorProcessor( ipVis );
				ipTable.copyBits( ipVis, i * w + w, j * h + h, Blitter.COPY );
				impTable.updateAndDraw();
			}
			catch ( final ImgLibException e )
			{
				IJ.log( "ImgLib2 Exception, vectors could not be painted." );
				e.printStackTrace();
			}
		}
		else
		{
			final Roi roi = new Roi( i * w + w, j * h + h, w, h );
			final Color c = IJ.getInstance().getForeground();
			ipTable.setColor( Color.WHITE );
			ipTable.fill( roi );
			ipTable.setColor( c );
		}
	}

	final static private FloatProcessor createMask( final ColorProcessor source )
	{
		final FloatProcessor mask = new FloatProcessor( source.getWidth(), source.getHeight() );
		final int maskColor = 0x0000ff00;
		final int[] sourcePixels = ( int[] )source.getPixels();
		final int n = sourcePixels.length;
		final float[] maskPixels = ( float[] )mask.getPixels();
		for ( int i = 0; i < n; ++i )
		{
			final int sourcePixel = sourcePixels[ i ] & 0x00ffffff;
			if ( sourcePixel == maskColor )
				maskPixels[ i ] = 0;
			else
				maskPixels[ i ] = 1;
		}
		return mask;
	}
	
	private static boolean setup( final TileSpecsImage entireImg )
	{
		new ImageJ();

		if ( entireImg == null )
		{
			IJ.error( "No image found" );
			return false;
		}
		
        final BoundingBox bbox = entireImg.getBoundingBox();
        if ( bbox.getDepth() < 2 )
        {
                IJ.error( "The image stack should contain at least two slices." );
                return false;
        }
		
		return true;
	}
	
	private static void run(
			final Params param,
			final TileSpecsImage entireImg,
			final int mipmapLevel )
	{
		if ( !setup( entireImg ) )
			return;

        final BoundingBox bbox = entireImg.getBoundingBox();
		int w = bbox.getWidth();
		int h = bbox.getHeight();
		while ( w > param.resolutionSpringMesh * minGridSize )
		{
			w /= 2;
			h /= 2;
		}

		final ImagePlus impTable;
		final ColorProcessor ipTable;
		final ColorProcessor[] renderedImages = new ColorProcessor[ bbox.getDepth() ];
		if ( exportDisplacementVectors )
		{
			ipTable = new ColorProcessor( w * bbox.getDepth() + w, h * bbox.getDepth() + h );

			final ColorProcessor ipScale = new ColorProcessor( w, h );
			final Color c = IJ.getInstance().getForeground();
			ipScale.setColor( Color.WHITE );
			ipScale.fill();
			ipScale.setColor( c );
			mpicbg.ij.util.Util.colorCircle( ipScale );

			ipTable.copyBits( ipScale, 0, 0, Blitter.COPY );

			for ( int i = 0; i < bbox.getDepth(); ++i )
			{
				renderedImages[ i ] = entireImg.render( i + bbox.getStartPoint().getZ(), mipmapLevel, param.layerScale );
				ColorProcessor ip = ( ColorProcessor )renderedImages[ i ].convertToRGB();
				while ( ip.getWidth() > w )
					ip = Downsampler.downsampleColorProcessor( ip );
				ipTable.copyBits( ip, i * w + w, 0, Blitter.COPY );
				ipTable.copyBits( ip, 0, i * h + h, Blitter.COPY );
			}
			impTable = new ImagePlus( "Block Matching Results", ipTable );
			impTable.show();
		}
		else
		{
			impTable = null;
			ipTable = null;
		}

		final SpringMesh mesh = new SpringMesh( param.resolutionSpringMesh, bbox.getWidth(), bbox.getHeight(), 1, 1000, 0.9f );
		final Collection< Vertex > vertices = mesh.getVertices();
		final RealPointSampleList< ARGBType > maskSamples = new RealPointSampleList< ARGBType >( 2 );
		for ( final Vertex vertex : vertices )
			maskSamples.add( new RealPoint( vertex.getL() ), new ARGBType( 0xffffffff ) );

		for ( int i = 0; i < bbox.getDepth(); ++i )
			for ( int j = i + 1; j < bbox.getDepth(); ++j )
			{
//				renderedImages[ i ] = entireImg.render( i + bbox.getStartPoint().getZ(), mipmapLevel );
//				renderedImages[ j ] = entireImg.render( j + bbox.getStartPoint().getZ(), mipmapLevel );
				final FloatProcessor ip1 = ( FloatProcessor )renderedImages[ i ].convertToFloat().duplicate();
				final FloatProcessor ip2 = ( FloatProcessor )renderedImages[ j ].convertToFloat().duplicate();

//				renderedImages[ i ] = entireImg.render( i + bbox.getStartPoint().getZ(), mipmapLevel );
//				renderedImages[ j ] = entireImg.render( j + bbox.getStartPoint().getZ(), mipmapLevel );

				final FloatProcessor ip1Mask = createMask( ( ColorProcessor )renderedImages[ i ].convertToRGB() );
				final FloatProcessor ip2Mask = createMask( ( ColorProcessor )renderedImages[ j ].convertToRGB() );
//				final FloatProcessor ip1Mask = null;
//				final FloatProcessor ip2Mask = null;

				final ArrayList< PointMatch > pm12 = match( param, bbox.getWidth(), bbox.getHeight(), ip1, ip2, ip1Mask, ip2Mask );
				IJ.log( i + " > " + j + " " + pm12.size() + " blockmatch candidates found." );

				if ( param.useLocalSmoothnessFilter )
				{
					filter( param, pm12 );
					IJ.log( pm12.size() + " blockmatch candidates passed local smoothness filter." );
				}
				if ( exportDisplacementVectors )
					display( param, bbox.getWidth(), bbox.getHeight(), pm12, maskSamples, impTable, ipTable, w, h, i, j );


				final ArrayList< PointMatch > pm21 = match( param, bbox.getWidth(), bbox.getHeight(), ip2, ip1, ip2Mask, ip1Mask );
				IJ.log( i + " < " + j + " " + pm21.size() + " blockmatch candidates found." );
				if ( param.useLocalSmoothnessFilter )
				{
					filter( param, pm21 );
					IJ.log( pm21.size() + " blockmatch candidates passed local smoothness filter." );
				}
				if ( exportDisplacementVectors )
					display( param, bbox.getWidth(), bbox.getHeight(), pm21, maskSamples, impTable, ipTable, w, h, j, i );
			}
		
		IJ.log( "Done" );
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
		
		final int mipmapLevel = 0;
		
		/* Set default block radius if not set by user */
		if ( params.blockRadius == -1 )
			params.blockRadius = params.searchRadius;
		
		/* Load all tilespecs into a 3d image stack */
		List< String > actualTileSpecFiles;
		if ( params.tileSpecFiles.size() == 1 )
			// It might be a non-json file that contains a list of
			actualTileSpecFiles = Utils.getListFromFile( params.tileSpecFiles.get( 0 ) );
		else
			actualTileSpecFiles = params.tileSpecFiles;
		final TileSpecsImage entireImg = TileSpecsImage.createImageFromFiles( actualTileSpecFiles );
		
		run( params, entireImg, mipmapLevel );
		
	}

	
}

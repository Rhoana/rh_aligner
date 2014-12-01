package org.janelia.alignment;

import ij.ImageJ;
import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.TreeMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.imageio.ImageIO;

import mpicbg.models.AffineModel2D;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.CoordinateTransformMesh;
import mpicbg.models.TransformMesh;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks.ImageProcessorWithMasks;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

public class RenderTiles {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--url", description = "URL to JSON tile spec", required = true )
        public String url;

        @Parameter( names = "--targetDir", description = "Directory to the output images", required = true )
        public String targetDir;
        
        @Parameter( names = "--res", description = " Mesh resolution, specified by the desired size of a triangle in pixels", required = false )
        public int res = 64;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
        @Parameter( names = "--scale", description = "scale factor applied to the target image (overrides --mipmap_level)", required = false )
        public double scale = -Double.NaN;
        
        @Parameter( names = "--quality", description = "JPEG quality float [0, 1]", required = false )
        public float quality = 0.85f;
        
        @Parameter( names = "--width", description = "Target image width (optional, if not given will be taken from the tilespecs)", required = false )
        public int width = -1;
        
        @Parameter( names = "--height", description = "Target image height (optional, if not given will be taken from the tilespecs)", required = false )
        public int height = -1;

        @Parameter( names = "--tileSize", description = "The size of a tile (one side of the square)", required = false )
        public int tileSize = 512;

	}

	
	
	private RenderTiles() {}
	
	
	final static Params parseParams( final String[] args )
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
        	jc.setProgramName( "java [-options] -cp render.jar + " + Render.class.getCanonicalName() );
        	jc.usage(); 
        	return null;
        }
		
		/* process params */
		if ( Double.isNaN( params.scale ) )
			params.scale = 1.0;
		
		return params;
	}	

	
	private static void saveAsTiles( 
			final ByteProcessor bpEntireSection,
			final int tileSize,
			final String outputDir,
			final int layer )
	{
		BufferedImage targetImage = new BufferedImage( tileSize, tileSize, BufferedImage.TYPE_BYTE_GRAY );
		WritableRaster origRaster = targetImage.getRaster();
		Object origData = origRaster.getDataElements( 0, 0, tileSize, tileSize, null );
		WritableRaster raster = targetImage.getRaster();
		
		
		bpEntireSection.snapshot();

		for ( int row = 0; row < bpEntireSection.getHeight(); row += tileSize )
		{
			int tileRowsNum = Math.min( row + tileSize, bpEntireSection.getHeight() ) - row;
			for ( int col = 0; col < bpEntireSection.getWidth(); col += tileSize )
			{
				int tileColsNum = Math.min( col + tileSize, bpEntireSection.getWidth() ) - col;

				// Crop the image
				bpEntireSection.resetRoi();
				bpEntireSection.setRoi( col, row, tileColsNum, tileRowsNum );
				ImageProcessor croppedImage = bpEntireSection.crop();

				// Save the cropped image 
				raster.setDataElements( 0, 0, tileColsNum, tileRowsNum, croppedImage.getPixels() );
								
				// Save the image to disk
				String outFile = outputDir + File.separatorChar + "tile_" + (row / tileSize) + "_" + (col / tileSize) + ".tif";
				Utils.saveImage( targetImage, outFile, "png" );
				
				raster.setDataElements( 0, 0, tileSize, tileSize, origData );
			}			
		}
	}



	public static void main( final String[] args )
	{

		final Params params = parseParams( args );
		
		if ( params == null )
			return;
		
		
		/* open tilespec */
		final URL url;
		final TileSpec[] tileSpecs;
		try
		{
			final Gson gson = new Gson();
			url = new URL( params.url );
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
		
		TileSpecsImage entireSection = new TileSpecsImage( tileSpecs, params.res );
		entireSection.setThreadsNum( params.numThreads );
		
		int width = -1;
		int height = -1;
		if ( ( params.width == -1 ) ||
			 ( params.height == -1 ) )
		{
			// Find the actual width and height of the image using the bounding boxes in the tilespecs
			BoundingBox bbox = entireSection.getBoundingBox();
			width = bbox.getWidth();
			height = bbox.getHeight();
		}
		
		if ( params.width != -1 )
			width = params.width;

		if ( params.height != -1 )
			height = params.height;

		// The mipmap level to work on
		// TODO: Should be a parameter from the user,
		//       and decide whether or not to create the mipmaps if they are missing
		int mipmapLevel = 0;

		// Render the entire image
		// TODO: maybe we should only render small tiles and save them (instead of rendering everything and saving the tiles)
		System.out.println( "Rendering the image to a temporary buffer" );
		ByteProcessor bpEntireSection = entireSection.render( tileSpecs[0].layer, mipmapLevel, ( float )params.scale, width, height );
		
		
		System.out.println( "Saving all tiles" );
		saveAsTiles( 
				bpEntireSection, 
				params.tileSize, 
				params.targetDir,
				tileSpecs[0].layer );
		
		System.out.println( "Done." );
		
	}

}

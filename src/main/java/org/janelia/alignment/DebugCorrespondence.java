package org.janelia.alignment;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.List;
import java.util.Random;

import mpicbg.models.Point;
import mpicbg.models.PointMatch;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

public class DebugCorrespondence {
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--inputfile", description = "The url of the correspondence json file", required = true )
        private String inputfile;

        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        private float layerScale = 0.1f;
        
        @Parameter( names = "--targetDir", description = "The output directory", required = true )
        private String targetDir;

        @Parameter( names = "--sifts", description = "Must be provided in case the correspondence is of sift featurs", required = false )
        private boolean sifts = false;

        @Parameter( names = "--showResults", description = "Whether to show the images that are created", required = false )
        private boolean showResults = false;

	}

	private DebugCorrespondence() {}

	
	private static void markAndSave(
			final String tileSpecUrl,
			final float scale,
			final int mipmapLevel,
			final List< PointMatch > candidates,
			final String targetPath,
			final int candidateNum,
			final boolean showResults )
	{
		final TileSpecsImage singleTileImage = TileSpecsImage.createImageFromFile( tileSpecUrl );
		final BoundingBox bbox = singleTileImage.getBoundingBox();
		final int layerIndex = bbox.getStartPoint().getZ();
		
		ByteProcessor bp = singleTileImage.render( layerIndex, mipmapLevel, scale );
		ColorProcessor cp = bp.convertToColorProcessor();
		
		System.out.println( "The output image size: " + cp.getWidth() + ", " + cp.getHeight() );
		final int width = cp.getWidth();
		final int height = cp.getHeight();

		// Mark the correspondence points
		int radius = (int)(Math.max( width, height ) * 0.005);
		radius = Math.max( radius, 10 );
		Random rng = new Random( 0 );
		int[] minMaxX = { Integer.MAX_VALUE, Integer.MIN_VALUE };
		int[] minMaxY = { Integer.MAX_VALUE, Integer.MIN_VALUE };
        for ( int i = 0; i < candidates.size(); i++ )
		{
			PointMatch pm = candidates.get( i );
			Point p = null;
			if ( candidateNum == 0 )
				p = pm.getP1();
			else if ( candidateNum == 1 )
				p = pm.getP2();
			
			int x = (int)(p.getW()[0] * scale);
			int y = (int)(p.getW()[1] * scale);

			minMaxX[0] = Math.min( minMaxX[0], x);
			minMaxX[1] = Math.max( minMaxX[1], x);
			minMaxY[0] = Math.min( minMaxY[0], y);
			minMaxY[1] = Math.max( minMaxY[1], y);

			Color curColor = new Color(
					Math.abs(rng.nextInt()) % 256,
					Math.abs(rng.nextInt()) % 256,
					Math.abs(rng.nextInt()) % 256 );
					
//			Color curColor = new Color(
//					255,
//					0,
//					0 );

			cp.setColor( curColor );
			int actualRadius = radius;
			if ( x - radius < 0 )
				actualRadius = Math.min(actualRadius, x - 1);
			if ( x + radius >= width )
				actualRadius = Math.min(actualRadius, width - x - 1);
			if ( y - radius < 0 )
				actualRadius = Math.min(actualRadius, y - 1);
			if ( y + radius >= height )
				actualRadius = Math.min(actualRadius, height - y - 1);

			// Cannot draw oval when dealing with large images (bug in ImageProcessor.fillOval)
//			System.out.println("Outputting oval at: " + (x - radius) + ", " + (y - radius));
//			cp.fillOval( x - radius, y - radius, 2 * radius, 2 * radius );
			System.out.println("Outputting square at: " + (x - actualRadius) + ", " + (y - actualRadius));
			Polygon po = new Polygon(
					new int[] { x - actualRadius, x + actualRadius, x + actualRadius, x - actualRadius },
					new int[] { y - actualRadius, y - actualRadius, y + actualRadius, y + actualRadius },
					4 );
			cp.fillPolygon(po);
		}
		
		final BufferedImage image = new BufferedImage( cp.getWidth(), cp.getHeight(), BufferedImage.TYPE_INT_RGB );
		final WritableRaster raster = image.getRaster();
		raster.setDataElements( 0, 0, cp.getWidth(), cp.getHeight(), cp.getPixels() );
		
		if ( showResults )
			new ImagePlus( "debug " + candidateNum, cp ).show();

		final BufferedImage targetImage = new BufferedImage( cp.getWidth(), cp.getHeight(), BufferedImage.TYPE_INT_RGB );
		final Graphics2D targetGraphics = targetImage.createGraphics();
		targetGraphics.drawImage( image, 0, 0, null );
		
		String fileName = targetPath + ".png";
		Utils.saveImage( targetImage, fileName, fileName.substring( fileName.lastIndexOf( '.' ) + 1 ) );
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

		final CorrespondenceSpec[] corr_data;
		try
		{
			final Gson gson = new Gson();
			URL url = new URL( params.inputfile );
			corr_data = gson.fromJson( new InputStreamReader( url.openStream() ), CorrespondenceSpec[].class );
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

		final String inFileName = params.inputfile.substring( params.inputfile.lastIndexOf( '/' ) + 1 );
		for ( int i = 0; i < corr_data.length; i++ )
		{
			System.out.println( "Saving debug data (before scaling) from corr_data entry " + i );
			
			final CorrespondenceSpec corrSpec = corr_data[i];
			
			final String tilespec1 = corrSpec.url1;
			final String tilespec2 = corrSpec.url2;
			
			// This index influences which correspondence group of points will belong
			// to which tilespec (either tilespec1 or tilespec2). This is because of
			// the way MatchLayersSiftFeatures is outputting its data
			int index = 0;
			if ( params.sifts )
				index = 1;
			
			// Save the first image
			final String outputFile2 = params.targetDir + File.separatorChar + inFileName + "_entry" + i + "_1";
			markAndSave( tilespec2, params.layerScale, mipmapLevel, corrSpec.correspondencePointPairs, outputFile2, 1 - index, params.showResults );

			// Save the second image
			final String outputFile1 = params.targetDir + File.separatorChar + inFileName + "_entry" + i + "_0";
			markAndSave( tilespec1, params.layerScale, mipmapLevel, corrSpec.correspondencePointPairs, outputFile1, index, params.showResults );
			
		}
		
		
		
		System.out.println( "Done." );
	}
}

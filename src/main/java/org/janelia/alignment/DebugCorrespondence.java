package org.janelia.alignment;

import ij.ImagePlus;
import ij.process.ColorProcessor;
import ij.process.ImageProcessor;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.InvertibleCoordinateTransform;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.TranslationModel2D;

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

        @Parameter( names = "--tsfile", description = "The url of the tilespec json file (of the montaged section)", required = true )
        private String tsfile;

        @Parameter( names = "--corrfile", description = "The url of the correspondence json file", required = true )
        private String corrfile;

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

        @Parameter( names = "--renderEntireTiles", description = "Whether to render the entire tiles or just the \"overlapping\" parts", required = false )
        private boolean renderEntireTiles = false;

	}

	private DebugCorrespondence() {}

	
	private static void markAndSave(
			final String imageUrl,
			final float scale,
			final int mipmapLevel,
			final List< PointMatch > candidates,
			final String targetPath,
			final int candidateNum,
			//final InvertibleCoordinateTransform revTransform,
			final boolean showResults,
			final boolean renderEntireTile )
	{

		if ( candidates.size() == 0 )
		{
			System.out.println( "No candidates found, skipping output" );
			return;
		}
		

//		final TileSpecsImage singleTileImage = TileSpecsImage.createImageFromFile( tileSpecUrl );
//		final BoundingBox bbox = singleTileImage.getBoundingBox();
//		final int layerIndex = bbox.getStartPoint().getZ();
		
		final ImagePlus imp = Utils.openImagePlusUrl( imageUrl );
		if ( imp == null )
		{
			System.err.println( "Failed to load image '" + imageUrl + "'." );
			return;
		}
		ColorProcessor cp = imp.getProcessor().convertToColorProcessor();
		final int width = imp.getWidth();
		final int height = imp.getHeight();
		
		// ColorProcessor cp = singleTileImage.render( layerIndex, mipmapLevel, scale );

		System.out.println( "The output image size: " + width + ", " + height );
		
		// Mark the correspondence points (0.5% radius)
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
			
			//revTransform.applyInPlace( p.getL() );
			
			int x = (int)p.getL()[0];
			int y = (int)p.getL()[1];

			minMaxX[0] = Math.min( minMaxX[0], x);
			minMaxX[1] = Math.max( minMaxX[1], x);
			minMaxY[0] = Math.min( minMaxY[0], y);
			minMaxY[1] = Math.max( minMaxY[1], y);
			
			Color curColor = new Color(
					Math.abs(rng.nextInt()) % 256,
					Math.abs(rng.nextInt()) % 256,
					Math.abs(rng.nextInt()) % 256 );
			
//			Color curColor = new Color(
//					0,
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
//			System.out.println("Outputting oval at: " + (x - actualRadius) + ", " + (y - actualRadius));
//			cp.fillOval( x - actualRadius, y - actualRadius, 2 * actualRadius, 2 * actualRadius );
			System.out.println("Outputting square at: " + (x - actualRadius) + ", " + (y - actualRadius));
			Polygon po = new Polygon(
					new int[] { x - actualRadius, x + actualRadius, x + actualRadius, x - actualRadius },
					new int[] { y - actualRadius, y - actualRadius, y + actualRadius, y + actualRadius },
					4 );
			cp.fillPolygon(po);
		}
		
		final BufferedImage image;
		if ( renderEntireTile )
		{
			image = new BufferedImage( cp.getWidth(), cp.getHeight(), BufferedImage.TYPE_INT_RGB );
			final WritableRaster raster = image.getRaster();
			raster.setDataElements( 0, 0, cp.getWidth(), cp.getHeight(), cp.getPixels() );
		}
		else
		{
			// Find the overlapping part of the image (1 of 4 strips on the "boundary" of the image)
			int fromX;
			int fromY;
			int outWidth;
			int outHeight;
			
			if ( minMaxX[1] - minMaxX[0] > minMaxY[1] - minMaxY[0] ) // horizontal strip
			{
				fromX = 0;
				outWidth = cp.getWidth();
				if ( ((double)minMaxY[0]) / cp.getHeight() > 0.5 ) // the Y points are closer to the bottom of the image
				{
					fromY = Math.max( 0, (int)( minMaxY[0] - ( cp.getHeight() * 0.1 ) ) ); // Add 10% to the height
					outHeight = cp.getHeight() - fromY;
				}
				else // the Y points are closer to the top of the image - upper strip
				{
					fromY = 0;
					outHeight = Math.min( cp.getHeight(), (int)( minMaxY[1] + ( cp.getHeight() * 0.1 ) ) ); // Add 10% to the height
				}
			}
			else // vertical strip
			{
				fromY = 0;
				outHeight = cp.getHeight();

				if ( ((double)minMaxX[0]) / cp.getWidth() > 0.5 ) // the X points are closer to the right side of the image
				{
					fromX = Math.max( 0, (int)( minMaxX[0] - ( cp.getWidth() * 0.1 ) ) ); // Add 10% to the width
					outWidth = cp.getWidth() - fromX;
				}
				else // the X points are closer to the left side of the image - left strip
				{
					fromX = 0;
					outWidth = Math.min( cp.getWidth(), (int)( minMaxX[1] + ( cp.getWidth() * 0.1 ) ) ); // Add 10% to the width
				}
			}

			// Crop the image
			cp.setRoi( fromX, fromY, outWidth, outHeight );
			ColorProcessor croppedImage = (ColorProcessor)cp.crop();
			
			// Set the cropped image as the output
			image = new BufferedImage( croppedImage.getWidth(), croppedImage.getHeight(), BufferedImage.TYPE_INT_RGB );
			final WritableRaster raster = image.getRaster();
			raster.setDataElements( 0, 0, croppedImage.getWidth(), croppedImage.getHeight(), croppedImage.getPixels() );

		}
		
		if ( showResults )
			new ImagePlus( "debug " + candidateNum, cp ).show();

		// Save the image to the disk
//		final BufferedImage targetImage = new BufferedImage( cp.getWidth(), cp.getHeight(), BufferedImage.TYPE_INT_RGB );
//		final Graphics2D targetGraphics = targetImage.createGraphics();
//		targetGraphics.drawImage( image, 0, 0, null );
//		
		String fileName = targetPath + ".jpg";
		Utils.saveImage( image, fileName, fileName.substring( fileName.lastIndexOf( '.' ) + 1 ) );
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

		// Read the tile spec file
		final TileSpec[] tileSpecs;
		try
		{
			final Gson gson = new Gson();
			final URL url = new URL( params.tsfile );
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

		
		// Create a map between an imageUrl and its corresponding reverse transformation
		/*
		final HashMap< String, InvertibleCoordinateTransform > imageToRevTransform = new HashMap< String, InvertibleCoordinateTransform >();
		for ( TileSpec ts : tileSpecs )
		{
			final String imageUrl = ts.getMipmapLevels().get( String.valueOf( mipmapLevel ) ).imageUrl;
			final CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
			final InvertibleCoordinateTransform revTransform = (( InvertibleCoordinateTransform )ctl.get(0)).createInverse();
			imageToRevTransform.put( imageUrl, revTransform );
		}
		*/
		
		final CorrespondenceSpec[] corr_data;
		try
		{
			final Gson gson = new Gson();
			URL url = new URL( params.corrfile );
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

		final String inFileName = params.corrfile.substring( params.corrfile.lastIndexOf( '/' ) + 1 );
		for ( int i = 0; i < corr_data.length; i++ )
		{
			System.out.println( "Saving debug data (before scaling) from corr_data entry " + i );
			
			final CorrespondenceSpec corrSpec = corr_data[i];
			
			final String imageUrl1 = corrSpec.url1;
			final String imageUrl2 = corrSpec.url2;
			
			// This index influences which correspondence group of points will belong
			// to which tilespec (either tilespec1 or tilespec2). This is because of
			// the way MatchLayersSiftFeatures is outputting its data
			int index = 0;
			if ( params.sifts )
				index = 1;
			
			// Save the first image
			final String outputFile2 = params.targetDir + File.separatorChar + inFileName + "_entry" + i + "_1";
			markAndSave( imageUrl2, params.layerScale, mipmapLevel, 
					corrSpec.correspondencePointPairs, outputFile2, 1 - index, 
					//imageToRevTransform.get( imageUrl2 ), 
					params.showResults, params.renderEntireTiles );

			// Save the second image
			final String outputFile1 = params.targetDir + File.separatorChar + inFileName + "_entry" + i + "_0";
			markAndSave( imageUrl1, params.layerScale, mipmapLevel, 
					corrSpec.correspondencePointPairs, outputFile1, index, 
					//imageToRevTransform.get( imageUrl1 ), 
					params.showResults, params.renderEntireTiles );
			
		}
		
		
		
		System.out.println( "Done." );
	}
}

package org.janelia.alignment;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.TreeMap;

import mpicbg.imagefeatures.Feature;
import mpicbg.models.SpringMesh;
import mpicbg.models.Vertex;

import org.janelia.alignment.FeatureSpec.ImageAndFeatures;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * Exports a mesh to a json file
 *
 */
public class ExportMesh {
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--imageWidth", description = "The width of the entire image (all layers), for consistent mesh computation", required = true )
        private int imageWidth;

        @Parameter( names = "--imageHeight", description = "The height of the entire image (all layers), for consistent mesh computation", required = true )
        private int imageHeight;

        @Parameter( names = "--targetPath", description = "The file to output the mesh (in json format)", required = true )
        public String targetPath;

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

	private ExportMesh() { }
	
	public static class MeshSpec
	{

		public static class Point
		{
			public int row;
			public int col;
			public float x;
			public float y;
			
			public Point( int row, int col, float y, float x ) {
				this.row = row;
				this.col = col;
				this.y = y;
				this.x = x;
			}
		}
		

		public int imageWidth;
		public int imageHeight;
		public int resolutionSpringMesh;
		public float stiffnessSpringMesh;
		public float maxStretchSpringMesh;
		public float dampSpringMesh;
		public float layerScale;
		
		public List< Point > points;
		
		public MeshSpec( int imageWidth, int imageHeight, int resolutionSpringMesh,
				float stiffnessSpringMesh, float maxStretchSpringMesh, float dampSpringMesh,
				float layerScale )
		{
			this.imageWidth = imageWidth;
			this.imageHeight = imageHeight;
			this.resolutionSpringMesh = resolutionSpringMesh;
			this.stiffnessSpringMesh = stiffnessSpringMesh;
			this.maxStretchSpringMesh = maxStretchSpringMesh;
			this.dampSpringMesh = dampSpringMesh;
			this.layerScale = layerScale;
		}

	}

	private static int roundFloat( float f )
	{
		final int decimalPlace = 3;
		/*
		BigDecimal bd = new BigDecimal( f );
	    bd = bd.setScale( decimalPlace, BigDecimal.ROUND_HALF_UP );
	    return bd.floatValue();
	    */
		int mul = 1;
		for ( int i = 0; i < decimalPlace; i++ )
			mul *= 10;
		return (int)(f * mul);
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
		
		// Create the mesh
		final int meshWidth = ( int )Math.ceil( params.imageWidth * params.layerScale );
		final int meshHeight = ( int )Math.ceil( params.imageHeight * params.layerScale );
		
		final SpringMesh singleMesh = new SpringMesh(
				params.resolutionSpringMesh,
				meshWidth,
				meshHeight,
				params.stiffnessSpringMesh,
				params.maxStretchSpringMesh * params.layerScale,
				params.dampSpringMesh ); 

		MeshSpec outSpec = new MeshSpec( params.imageWidth, params.imageHeight,
				params.resolutionSpringMesh, params.stiffnessSpringMesh,
				params.maxStretchSpringMesh, params.dampSpringMesh,
				params.layerScale );

		System.out.println( "Transforming points" );

		// Export the mesh to json format
		ArrayList< Vertex > allVertices = singleMesh.getVertices();
		// find the number of different x values, and the number of different y values
		HashSet< Integer > xValues = new HashSet< Integer >();
		HashSet< Integer > yValues = new HashSet< Integer >();
		for ( Vertex v : allVertices ) {
			xValues.add( roundFloat( v.getL()[ 0 ] ) );
			yValues.add( roundFloat( v.getL()[ 1 ] ) );
		}
		
		
		// Sort the x and y values (so we'll have a mapping between a flaot value and its index)
		Integer[] xValuesArr = xValues.toArray( new Integer[0] );
		Integer[] yValuesArr = yValues.toArray( new Integer[0] );
		Arrays.sort( xValuesArr );
		Arrays.sort( yValuesArr );
		
		HashMap< Integer, Integer > xValuesMap = new HashMap< Integer, Integer >();
		HashMap< Integer, Integer > yValuesMap = new HashMap< Integer, Integer >();
		
		for ( int i = 0; i < xValuesArr.length; i++ ) {
			xValuesMap.put( xValuesArr[ i ], i );
		}
		for ( int i = 0; i < yValuesArr.length; i++ ) {
			yValuesMap.put( yValuesArr[ i ], i );
		}
		
		outSpec.points = new ArrayList< MeshSpec.Point >();
		for ( Vertex v : allVertices ) {
			int x = roundFloat( v.getL()[ 0 ] );
			int y = roundFloat( v.getL()[ 1 ] );
			outSpec.points.add( new MeshSpec.Point(
					yValuesMap.get( y ),
					xValuesMap.get( x ),
					v.getL()[ 1 ],
					v.getL()[ 0 ] ) );
		}
		
		
		
		// Export the file
		try {
			Writer writer = new FileWriter(params.targetPath);
	        //Gson gson = new GsonBuilder().create();
	        Gson gson = new GsonBuilder().setPrettyPrinting().create();
	        gson.toJson(outSpec, writer);
	        writer.close();
	    }
		catch ( final IOException e )
		{
			System.err.println( "Error writing JSON file: " + params.targetPath );
			e.printStackTrace( System.err );
		}

		System.out.println( "Done" );
	}
	
}

/**
e * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package org.janelia.alignment;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Writer;
import java.io.FileWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import mpicbg.trakem2.transform.AffineModel2D;
import mpicbg.trakem2.transform.MovingLeastSquaresTransform;

import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.PointMatch;
import mpicbg.models.Spring;
import mpicbg.models.SpringMesh;
import mpicbg.models.Tile;
import mpicbg.models.TileConfiguration;
import mpicbg.models.Vertex;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

/** 
 * @author Seymour Knowles-Barley
 */
public class OptimizeSeriesElastic
{
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--inputfile", description = "Correspondence list file (correspondences are in world space, after application of any existing transformations)", required = true )
        private String inputfile;

        @Parameter( names = "--modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int modelIndex = 3;
        
        @Parameter( names = "--meshWidth", description = "Mesh width (in pixels) for all images in this series.", required = true )
        public int meshWidth;
        
        @Parameter( names = "--meshHeight", description = "Mesh height (in pixels) for all images in this series.", required = true )
        public int meshHeight;
        
        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        private float layerScale = 0.2f;
        
        @Parameter( names = "--resolutionSpringMesh", description = "resolutionSpringMesh", required = false )
        private int resolutionSpringMesh = 32;
        
        @Parameter( names = "--springLengthSpringMesh", description = "springLengthSpringMesh", required = false )
        private float springLengthSpringMesh = 100f;
		
        @Parameter( names = "--stiffnessSpringMesh", description = "stiffnessSpringMesh", required = false )
        private float stiffnessSpringMesh = 0.1f;
		
        @Parameter( names = "--dampSpringMesh", description = "dampSpringMesh", required = false )
        private float dampSpringMesh = 0.9f;
		
        @Parameter( names = "--maxStretchSpringMesh", description = "maxStretchSpringMesh", required = false )
        private float maxStretchSpringMesh = 2000.0f;
        
        @Parameter( names = "--maxEpsilon", description = "maxEpsilon", required = false )
        private float maxEpsilon = 25.0f;
        
        @Parameter( names = "--maxIterationsSpringMesh", description = "maxIterationsSpringMesh", required = false )
        private int maxIterationsSpringMesh = 1000;
        
        @Parameter( names = "--maxPlateauwidthSpringMesh", description = "maxPlateauwidthSpringMesh", required = false )
        private int maxPlateauwidthSpringMesh = 200;
        
        @Parameter( names = "--resolutionOutput", description = "resolutionOutput", required = false )
        private int resolutionOutput = 128;
        
        @Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
	}
	
	private OptimizeSeriesElastic() {}
	
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

		// The mipmap level to work on
		// TODO: Should be a parameter from the user,
		//       and decide whether or not to create the mipmaps if they are missing
		int mipmapLevel = 0;


		/* Elastic alignment */

		/* Initialization */
		final TileConfiguration initMeshes = new TileConfiguration();

		final HashMap< String, Tile< ? > > tilesMap = new HashMap< String, Tile< ? > >();
		//final HashMap< String, Tile< ? > > fixedTilesMap = new HashMap< String, Tile< ? > >();
		final HashMap< String, SpringMesh > tileMeshMap = new HashMap< String, SpringMesh >();
		//final ArrayList< SpringMesh > meshes = new ArrayList< SpringMesh >();
		
		for ( final CorrespondenceSpec corr : corr_data )
		{
			// Here we generate a new mesh based on the overall (command line) width / height.
			// If section images are different sizes, this should be the extended bounding box for all sections.
			if (!tileMeshMap.containsKey( corr.url1 ))
			{
				final SpringMesh mesh = Utils.getMesh( params.meshWidth, params.meshHeight, params.layerScale, params.resolutionSpringMesh, params.stiffnessSpringMesh, params.dampSpringMesh, params.maxStretchSpringMesh );
				tileMeshMap.put( corr.url1, mesh );
				Tile< ? > t = Utils.createTile(params.modelIndex);
				tilesMap.put(corr.url1, t);
			}
			if (!tileMeshMap.containsKey( corr.url2 ))
			{
				final SpringMesh mesh = Utils.getMesh( params.meshWidth, params.meshHeight, params.layerScale, params.resolutionSpringMesh, params.stiffnessSpringMesh, params.dampSpringMesh, params.maxStretchSpringMesh );
				tileMeshMap.put( corr.url2, mesh );
				Tile< ? > t = Utils.createTile(params.modelIndex);
				tilesMap.put(corr.url2, t);
			}
			
			Tile < ? > tileA = tilesMap.get( corr.url1 );
			Tile < ? > tileB = tilesMap.get( corr.url2 );
			
//            if (layer1Fixed)
//            {
//                initMeshes.fixTile( tileA );
//            }
//			  else // Link tiles
            
			// Link tiles
			if (tileA != null && tileB != null && corr.correspondencePointPairs.size() > tileA.getModel().getMinNumMatches())
			{
				initMeshes.addTile(tileA);
				initMeshes.addTile(tileB);
				tileA.connect(tileB, corr.correspondencePointPairs);
			}
			
			SpringMesh meshA = tileMeshMap.get(corr.url1);
			SpringMesh meshB = tileMeshMap.get(corr.url2);
			
			// Link meshes
			for (PointMatch pm : corr.correspondencePointPairs)
			{
				PointMatch closestPointA = meshA.findClosestTargetPoint(pm.getP1().getW());
				float dx = closestPointA.getP1().getW()[0] - pm.getP1().getW()[0]; 
				float dy = closestPointA.getP1().getW()[1] - pm.getP1().getW()[1];
				
				final Vertex p1 = ( Vertex )closestPointA.getP1();
				float p2x = pm.getP2().getW()[0] + dx;
				float p2y = pm.getP2().getW()[1] + dy;
				final Vertex p2 = new Vertex( new float[]{p2x, p2y} );
				p1.addSpring( p2, new Spring( 0, 1.0f ) );
				meshB.addPassiveVertex( p2 );
				
			}
			
			System.out.println(corr.url1 + " -> " + corr.url2 + ": added " + corr.correspondencePointPairs.size() + " links.");
			
		}
		
		/* pre-align by optimizing a piecewise linear model */
		try
		{
			initMeshes.optimize(
					params.maxEpsilon,
					params.maxIterationsSpringMesh,
					params.maxPlateauwidthSpringMesh );
		}
		catch ( final Exception e )
		{
			System.out.println( "Exception while optimizing a piecewise linear model:" );
			e.printStackTrace();
			return;
		}
		
		for ( final Map.Entry< String, SpringMesh > entry : tileMeshMap.entrySet() ) 
		{
			final SpringMesh mesh = entry.getValue();
			final Tile < ? > tile = tilesMap.get( entry.getKey() );
			mesh.init( tile.getModel() );
		}
		
		final ArrayList< SpringMesh > meshes = new ArrayList< SpringMesh >(tileMeshMap.values());
		
		/* optimize the meshes */
		try
		{
			final long t0 = System.currentTimeMillis();
			System.out.println("Optimizing spring meshes...");

			SpringMesh.optimizeMeshes(
					meshes,
					params.maxEpsilon,
					params.maxIterationsSpringMesh,
					params.maxPlateauwidthSpringMesh,
					false );

			System.out.println( "Done optimizing spring meshes. Took " + ( System.currentTimeMillis() - t0 ) + " ms" );

		}
		catch ( final NotEnoughDataPointsException e )
		{
			System.out.println( "There were not enough data points to get the spring mesh optimizing." );
			e.printStackTrace();
			return;
		}

		System.out.println( "Optimization complete. Generating tile transforms.");
		
		ArrayList< TileSpec > out_tiles = new ArrayList< TileSpec >();
		
		/* calculate bounding box */
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

		/* translate relative to bounding box */
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
		
		for ( final Map.Entry< String, SpringMesh > entry : tileMeshMap.entrySet() ) 
		{
			final String tileUrl = entry.getKey();
			final SpringMesh mesh = entry.getValue();
			final TileSpec ts = new TileSpec();
			ts.setMipmapLevelImageUrl("" + mipmapLevel, tileUrl);
			ts.width = fullWidth;
			ts.height = fullHeight;

			// bounding box after transformations are applied [left, right, top, bottom] possibly with extra entries for [front, back, etc.]
			final float[] meshMin = new float[ 2 ];
			final float[] meshMax = new float[ 2 ];
			mesh.bounds( meshMin, meshMax );			
			ts.bbox = new float[] {meshMin[0], meshMax[0], meshMin[1], meshMax[1]};
			

			try
			{
				final MovingLeastSquaresTransform mlt = new MovingLeastSquaresTransform();
				mlt.setModel( AffineModel2D.class );
				mlt.setAlpha( 2.0f );
				mlt.setMatches( mesh.getVA().keySet() );
	
			    Transform addedTransform = new Transform();				    
			    addedTransform.className = mlt.getClass().getCanonicalName().toString();
			    addedTransform.dataString = mlt.toDataString();
				ts.transforms = new Transform[] {addedTransform};
			}
			catch ( final Exception e )
			{
				System.out.println( "Error applying transform to tile " + tileUrl + "." );
				e.printStackTrace();
			}

			// Image output and visualization code
			
//			final CoordinateTransformMesh mltMesh = new CoordinateTransformMesh( mlt, params.resolutionOutput, params.meshWidth, params.meshHeight );
//			final TransformMeshMapping< CoordinateTransformMesh > mltMapping = new TransformMeshMapping< CoordinateTransformMesh >( mltMesh );
//
//			final ImageProcessor source, target;
//			if ( p.rgbWithGreenBackground )
//			{
//				target = new ColorProcessor( width, height );
//				for ( int j = width * height - 1; j >=0; --j )
//					target.set( j, 0xff00ff00 );
//				source = stack.getProcessor( slice ).convertToRGB();
//			}
//			else
//			{
//				target = stack.getProcessor( slice ).createProcessor( width, height );
//				source = stack.getProcessor( slice );
//			}
//
//			if ( p.interpolate )
//			{
//				mltMapping.mapInterpolated( source, target );
//			}
//			else
//			{
//				mltMapping.map( source, target );
//			}
//			final ImagePlus impTarget = new ImagePlus( "elastic mlt " + i, target );
//			if ( p.visualize )
//			{
//				final Shape shape = mltMesh.illustrateMesh();
//				impTarget.setOverlay( shape, IJ.getInstance().getForeground(), new BasicStroke( 1 ) );
//			}
//			IJ.save( impTarget, p.outputPath + "elastic-" + String.format( "%05d", i ) + ".tif" );
			
		}

		System.out.println( "Exporting tiles.");
		
		try {
			Writer writer = new FileWriter(params.targetPath);
	        //Gson gson = new GsonBuilder().create();
	        Gson gson = new GsonBuilder().setPrettyPrinting().create();
	        gson.toJson(out_tiles, writer);
	        writer.close();
	    }
		catch ( final IOException e )
		{
			System.err.println( "Error writing JSON file: " + params.targetPath );
			e.printStackTrace( System.err );
		}

		System.out.println( "Done." );
	}
	
}

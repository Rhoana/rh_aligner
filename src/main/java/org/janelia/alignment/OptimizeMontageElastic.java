/**
 * License: GPL
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import mpicbg.models.AffineModel2D;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.MovingLeastSquaresTransform2;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.PointMatch;
import mpicbg.models.Spring;
import mpicbg.models.SpringMesh;
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
public class OptimizeMontageElastic
{
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--inputfile", description = "Correspondence list file (correspondences are in world space, after application of any existing transformations)", required = true )
        private String inputfile;
        
        @Parameter( names = "--tilespecfile", description = "Tilespec file containing all tiles for this montage and current transforms", required = true )
        private String tilespecfile;
        
        @Parameter( names = "--modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int modelIndex = 3;
        
        @Parameter( names = "--tileWidth", description = "Tile width (specify if same for all tiles, otherwise tilespec data will be used)", required = false )
        private double tileWidth = 0;
        
        @Parameter( names = "--tileHeight", description = "Tile height (specify if same for all tiles, otherwise tilespec data will be used)", required = false )
        private double tileHeight = 0;
        
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
        
		@Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
	}
	
	private OptimizeMontageElastic() {}
	
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
		
		/* Initialization */
		/* read all tilespecs */
		final HashMap< String, TileSpec > tileSpecMap = new HashMap< String, TileSpec >();
		
		/* open tilespec */
		final URL url;
		final TileSpec[] tileSpecs;
		try
		{
			final Gson gson = new Gson();
			url = new URL( params.tilespecfile );
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
		
		for (TileSpec ts : tileSpecs)
		{
			tileSpecMap.put(ts.imageUrl, ts);
		}
		
		/* create tiles and models for all patches */
		//final HashMap< String, Tile< ? > > tilesMap = new HashMap< String, Tile< ? > >();
		//final ArrayList< Tile< ? > > fixedTiles = new ArrayList< Tile< ? > > ();
		
		final HashMap< String, TileSpec > fixedTilesMap = new HashMap< String, TileSpec > ();
		
//		/* make pairwise global models local */
//		final ArrayList< Triple< Tile< ? >, Tile< ? >, InvertibleCoordinateTransform > > pairs =
//			new ArrayList< Triple< Tile< ? >, Tile< ? >, InvertibleCoordinateTransform > >();

		final HashMap< String, SpringMesh > tileMeshMap = new HashMap< String, SpringMesh >();
		
		/*
		 * The following casting madness is necessary to get this code compiled
		 * with Sun/Oracle Java 6 which otherwise generates an inconvertible
		 * type exception.
		 * 
		 * TODO Remove as soon as this bug is fixed in Sun/Oracle javac.
		 */
		for ( final CorrespondenceSpec corr : corr_data )
		{
			// Here we generate a new mesh based on the tile width / height (fixed or read from tilespec).
			if (!tileMeshMap.containsKey( corr.url1 ))
			{
				final TileSpec ts = tileSpecMap.get(corr.url1);
				final CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
				final SpringMesh mesh = Utils.getMesh( ts.width, ts.height, params.layerScale, params.resolutionSpringMesh, params.stiffnessSpringMesh, params.dampSpringMesh, params.maxStretchSpringMesh );

				// Apply the tilespec transform to the mesh
				mesh.init(ctl);
				
				tileMeshMap.put( corr.url1, mesh );
			}
			if (!tileMeshMap.containsKey( corr.url2 ))
			{
				final TileSpec ts = tileSpecMap.get(corr.url2);
				final CoordinateTransformList< CoordinateTransform > ctl = ts.createTransformList();
				final SpringMesh mesh = Utils.getMesh( ts.width, ts.height, params.layerScale, params.resolutionSpringMesh, params.stiffnessSpringMesh, params.dampSpringMesh, params.maxStretchSpringMesh );

				// Apply the tilespec transform to the mesh
				mesh.init(ctl);
				
				tileMeshMap.put( corr.url2, mesh );
			}
			
			SpringMesh meshA = tileMeshMap.get(corr.url1);
			SpringMesh meshB = tileMeshMap.get(corr.url2);
			
			// Link meshes
			for (PointMatch pm : corr.correspondencePointPairs)
			{
				Vertex closestVertexA = meshA.findClosestTargetVertex(pm.getP1().getW());
				float dx = closestVertexA.getW()[0] - pm.getP1().getW()[0]; 
				float dy = closestVertexA.getW()[1] - pm.getP1().getW()[1];

				float p2x = pm.getP2().getW()[0] + dx;
				float p2y = pm.getP2().getW()[1] + dy;
				final Vertex p2 = new Vertex( new float[]{p2x, p2y} );
				closestVertexA.addSpring( p2, new Spring( 0, 1.0f ) );
				meshB.addPassiveVertex( p2 );
				
			}
			
			System.out.println(corr.url1 + " -> " + corr.url2 + ": added " + corr.correspondencePointPairs.size() + " links.");
			
		}

		final ArrayList< SpringMesh > meshes = new ArrayList< SpringMesh >(tileMeshMap.values());
		
		/* optimize the meshes */
		try
		{
			final long t0 = System.currentTimeMillis();
			System.out.println( "Optimizing spring meshes..." );

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
		
		/* apply */
		for ( final Map.Entry< String, SpringMesh > entry : tileMeshMap.entrySet() ) 
		{
			final String tileUrl = entry.getKey();
			final SpringMesh mesh = entry.getValue();
			final TileSpec ts = tileSpecMap.get(tileUrl);
			if ( !fixedTilesMap.containsKey( tileUrl ) )
			{
				if (mesh.getVA() == null)
				{
					System.out.println( "Error generating transform for tile " + tileUrl + "." );
					continue;
				}
				final Set< PointMatch > matches = mesh.getVA().keySet();

//				/* compensate for existing coordinate transform bounding box */
//				for ( final PointMatch pm : matches )
//				{
//					final Point p1 = pm.getP1();
//					final float[] l = p1.getL();
//					l[ 0 ] += box.x;
//					l[ 1 ] += box.y;
//				}

				try
				{
					//Generate the transform for this mesh
					final MovingLeastSquaresTransform2 mlt = new MovingLeastSquaresTransform2();
					mlt.setModel( AffineModel2D.class );
					mlt.setAlpha( 2.0f );
					mlt.setMatches( matches );
					
					//Apply to the corresponding tilespec transforms
					ArrayList< Transform > outTransforms = new ArrayList< Transform >(Arrays.asList(ts.transforms));
				    Transform addedTransform = new Transform();				    
				    addedTransform.className = mlt.getClass().toString();
				    addedTransform.dataString = mlt.toString();
					outTransforms.add(addedTransform);
					ts.transforms = outTransforms.toArray(ts.transforms);
					out_tiles.add(ts);
				}
				catch ( final Exception e )
				{
					System.out.println( "Error applying transform to tile " + tileUrl + "." );
					e.printStackTrace();
				}
				
//				patch.appendCoordinateTransform( mlt );
//				box = patch.getCoordinateTransformBoundingBox();
//
//				patch.getAffineTransform().setToTranslation( box.x, box.y );
//				patch.updateInDatabase( "transform" );
//				patch.updateBucket();
//
//				patch.updateMipMaps();
			}
		}

		System.out.println( "Exporting " + out_tiles.size() + " tiles.");
		
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

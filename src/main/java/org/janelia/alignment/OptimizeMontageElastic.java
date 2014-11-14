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
import java.util.List;
import java.util.Map;
import java.util.Set;

import mpicbg.trakem2.transform.AffineModel2D;
import mpicbg.trakem2.transform.MovingLeastSquaresTransform2;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
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

        @Parameter( names = "--corrfile", description = "Correspondence list file (correspondences are in world space, after application of any existing transformations)", required = true )
        private String corrfile;
        
        @Parameter( names = "--tilespecfile", description = "Tilespec file containing all tiles for this montage and current transforms", required = true )
        private String tilespecfile;

        @Parameter( names = "--fixedTiles", description = "Fixed tiles indices (space separated)", variableArity = true, required = true )
        public List<Integer> fixedTiles = new ArrayList<Integer>();
        
        @Parameter( names = "--modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int modelIndex = 3;
        
        @Parameter( names = "--layerScale", description = "Layer scale", required = false )
        private float layerScale = 0.5f;
        
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
        
        @Parameter( names = "--useLegacyOptimizer", description = "Use legacy optimizer", required = false )
        private boolean useLegacyOptimizer = false;

		@Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
	}
	
	private OptimizeMontageElastic() {}
	
	/* Returns a map between a tile idx -> map of another tile index and the correspondece points between them */
	private static HashMap< Integer, HashMap< Integer, CorrespondenceSpec > > parseCorrespondenceFile(
			final String corrFileUrl,
			final HashMap< String, Integer > imgUrlToTileIdxs )
	{
		System.out.println( "Parsing correspondence file" );
		HashMap< Integer, HashMap< Integer, CorrespondenceSpec > > tilesCorrs = new HashMap<Integer, HashMap<Integer,CorrespondenceSpec>>();

		// Open and parse the json file
		final CorrespondenceSpec[] corr_data;
		try
		{
			final Gson gson = new Gson();
			URL url = new URL( corrFileUrl );
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
			final int tile1Idx = imgUrlToTileIdxs.get( corr.url1 );
			final int tile2Idx = imgUrlToTileIdxs.get( corr.url2 );
			final HashMap< Integer, CorrespondenceSpec > innerMapping;

			if ( tilesCorrs.containsKey( tile1Idx ) )
			{
				innerMapping = tilesCorrs.get( tile1Idx );
			}
			else
			{
				innerMapping = new HashMap<Integer, CorrespondenceSpec>();
				tilesCorrs.put( tile1Idx, innerMapping );
			}
			// Assuming that no two files have the same correspondence spec url values
			innerMapping.put( tile2Idx,  corr );
		}

		return tilesCorrs;
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

	private static List< SpringMesh > fixAllPointMatchVertices(
			final Params param,
			final TileSpec[] tileSpecs,
			final HashMap< Integer, HashMap< Integer, CorrespondenceSpec > > tilesCorrs )
	{

		System.out.println( "Fixing point matches " );
		final List< SpringMesh > meshes = Utils.createMeshes( tileSpecs, 
				param.springLengthSpringMesh, param.stiffnessSpringMesh, param.maxStretchSpringMesh,
				param.layerScale, param.dampSpringMesh );
		for ( int i = 0; i < meshes.size(); ++i )
		{
			if ( tilesCorrs.containsKey( i ) )
			{
				final SpringMesh singleMesh = meshes.get( i );

				HashMap< Integer, CorrespondenceSpec > tileICorrs = tilesCorrs.get( i );
				for ( CorrespondenceSpec corrspec : tileICorrs.values() )
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
		
		/* Initialization */
		/* read all tilespecs */
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
		
		final HashMap< String, Integer > imgUrlToTileIdxs = new HashMap<String, Integer>();
		for ( int i = 0; i < tileSpecs.length; i++ )
		{
			TileSpec ts = tileSpecs[ i ];
			String imageUrl = ts.getMipmapLevels().get("" + mipmapLevel).imageUrl;
			imgUrlToTileIdxs.put(imageUrl, i);
		}

		HashMap< Integer, HashMap< Integer, CorrespondenceSpec > > tilesCorrs = 
				parseCorrespondenceFile( params.corrfile, imgUrlToTileIdxs );
		
		final List< SpringMesh > meshes = 
				fixAllPointMatchVertices( params, tileSpecs, tilesCorrs );
		
		System.out.println( "Creating springs" );
		// Iterate over all pairs of tile corrs, and create the appropriate matches
		for ( int tile1Idx = 0; tile1Idx < tileSpecs.length; tile1Idx++ )
		{
			for ( int tile2Idx = tile1Idx + 1; tile2Idx < tileSpecs.length; tile2Idx++ )
			{
                final CorrespondenceSpec corrspec12;
                final List< PointMatch > pm12;
                final CorrespondenceSpec corrspec21;
                final List< PointMatch > pm21;

                final SpringMesh m1 = meshes.get( tile1Idx );
                final SpringMesh m2 = meshes.get( tile2Idx );
                
                if ( !tilesCorrs.containsKey( tile1Idx ) || !tilesCorrs.get( tile1Idx ).containsKey( tile2Idx ) )
                {
                	corrspec12 = null;
                	pm12 = new ArrayList< PointMatch >();
                }
                else
                {
                	corrspec12 = tilesCorrs.get( tile1Idx ).get( tile2Idx );
                	pm12 = corrspec12.correspondencePointPairs;
                }

                if ( !tilesCorrs.containsKey( tile2Idx ) || !tilesCorrs.get( tile2Idx ).containsKey( tile1Idx ) )
                {
                	corrspec21 = null;
                	pm21 = new ArrayList< PointMatch >();
                }
                else
                {
                	corrspec21 = tilesCorrs.get( tile2Idx ).get( tile1Idx );
                	pm21 = corrspec21.correspondencePointPairs;
                }

                for ( final PointMatch pm : pm12 )
                {
                	final Vertex p1 = ( Vertex )pm.getP1();
                	final Vertex p2 = new Vertex( pm.getP2() );
                	p1.addSpring( p2, new Spring( 0, 1.0f ) );
                	m2.addPassiveVertex( p2 );
                }

                for ( final PointMatch pm : pm21 )
                {
                	final Vertex p1 = ( Vertex )pm.getP1();
                	final Vertex p2 = new Vertex( pm.getP2() );
                	p1.addSpring( p2, new Spring( 0, 1.0f ) );
                	m1.addPassiveVertex( p2 );
                }
				
			}
		}
		
		
		/* initialize */
		for ( int i = 0; i < tileSpecs.length; i++ )
		{
			SpringMesh m = meshes.get( i );
			m.init( tileSpecs[i].createTransformList() );
		}
		
		/* optimize the meshes */
		try
		{
			final long t0 = System.currentTimeMillis();
			System.out.println( "Optimizing spring meshes..." );

			if ( params.useLegacyOptimizer )
			{
				System.out.println( "  ...using legacy optimizer...");
				SpringMesh.optimizeMeshes2(
						meshes,
						params.maxEpsilon,
						params.maxIterationsSpringMesh,
						params.maxPlateauwidthSpringMesh );
			}
			else
			{
				SpringMesh.optimizeMeshes(
						meshes,
						params.maxEpsilon,
						params.maxIterationsSpringMesh,
						params.maxPlateauwidthSpringMesh );
			}
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
		for ( int i = 0; i < tileSpecs.length; i++ )
		{
			if ( params.fixedTiles.contains( i ) )
			{
				// Fixed tile, nothing to change in the tilespec
				final TileSpec ts = tileSpecs[ i ];
				out_tiles.add(ts);
			}
			else
			{
				final SpringMesh mesh = meshes.get( i );
				final TileSpec ts = tileSpecs[ i ];
				
				if (mesh.getVA() == null)
				{
					System.out.println( "Error generating transform for tile " + i + "." );
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
				    Transform addedTransform = new Transform();
				    addedTransform.className = mlt.getClass().getCanonicalName();
				    addedTransform.dataString = mlt.toDataString();
				    
					//ArrayList< Transform > outTransforms = new ArrayList< Transform >(Arrays.asList(ts.transforms));
					// (override previous transformations)
					ArrayList< Transform > outTransforms = new ArrayList< Transform >( );
					outTransforms.add(addedTransform);
					ts.transforms = outTransforms.toArray(ts.transforms);
					out_tiles.add(ts);
				}
				catch ( final Exception e )
				{
					System.out.println( "Error applying transform to tile " + i + "." );
					e.printStackTrace();
				}

			}
		}
		

		System.out.println( "Exporting " + out_tiles.size() + " tiles.");
		
		try {
			Writer writer = new FileWriter( params.targetPath );
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

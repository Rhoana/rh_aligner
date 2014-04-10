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
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import mpicbg.models.Model;
import mpicbg.models.PointMatch;
import mpicbg.models.Tile;
import mpicbg.models.TileConfiguration;
import mpicbg.trakem2.transform.AffineModel2D;
import mpicbg.trakem2.transform.HomographyModel2D;
import mpicbg.trakem2.transform.RigidModel2D;
import mpicbg.trakem2.transform.SimilarityModel2D;
import mpicbg.trakem2.transform.TranslationModel2D;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

/**
 * @author Seymour Knowles-Barley
 */
public class OptimizeMontageTransform
{
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--inputfile", description = "Correspondence list file", required = true )
        private String inputfile;
                        
        @Parameter( names = "--modelIndex", description = "Model Index: 0=Translation, 1=Rigid, 2=Similarity, 3=Affine, 4=Homography", required = false )
        private int modelIndex = 1;
                        
        @Parameter( names = "--filterOutliers", description = "Filter outliers during optimization", required = false )
        private boolean filterOutliers = false;
                        
        @Parameter( names = "--maxEpsilon", description = "Max epsilon", required = false )
        private float maxEpsilon = 100.0f;
                        
        @Parameter( names = "--maxIterations", description = "Max iterations", required = false )
        private int maxIterations = 2000;
        
        @Parameter( names = "--maxPlateauwidth", description = "Max plateau width", required = false )
        private int maxPlateauwidth = 200;
                                
        @Parameter( names = "--meanFactor", description = "Mean factor", required = false )
        private float meanFactor = 3.0f;
                        
        @Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();
        
	}
	
	private OptimizeMontageTransform() {}
	
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
		
//		final boolean tilesAreInPlace = true;
		
		final Map< String, Tile< ? > > tilesMap = new HashMap< String, Tile< ? > >();
//		final List< Tile< ? > > tiles = new ArrayList< Tile< ? > >();
		final List< Tile< ? > > fixedTiles = new ArrayList< Tile< ? > >();
//		final List< Tile< ? >[] > tilePairs = new ArrayList< Tile< ? >[] >();
		
		for (CorrespondenceSpec corr : corr_data)
		{
			final Tile< ? > tile1;
			final Tile< ? > tile2;
			
			if (tilesMap.containsKey(corr.url1))
			{
				tile1 = tilesMap.get(corr.url1);
			}
			else
			{
				tile1 = Utils.createTile( params.modelIndex );
				tilesMap.put(corr.url1, tile1);
				//tiles.add(tile1);
			}
			
			if (tilesMap.containsKey(corr.url2))
			{
				tile2 = tilesMap.get(corr.url2);
			}
			else
			{
				tile2 = Utils.createTile( params.modelIndex );
				tilesMap.put(corr.url2, tile2);
				//tiles.add(tile2);
			}
			tile1.addConnectedTile(tile2);
			tile2.addConnectedTile(tile1);

			// Forward Point Matches
			tile1.addMatches( corr.correspondencePointPairs );
			
			// Backward Point Matches
			for ( PointMatch pm : corr.correspondencePointPairs )
			{
				tile2.addMatch(new PointMatch(pm.getP2(), pm.getP1()));
				System.out.println("p1 " + pm.getP1().getW()[0] + ", " + pm.getP1().getW()[1]);
				System.out.println("p2 " + pm.getP2().getW()[0] + ", " + pm.getP2().getW()[1]);
			}
		}
		
//		final List< Set< Tile< ? > > > graphs = AbstractAffineTile2D.identifyConnectedGraphs( tiles );
//
//		final List< AbstractAffineTile2D< ? > > interestingTiles;
//		if ( largestGraphOnlyIn )
//		{
//			/* find largest graph. */
//
//			Set< Tile< ? > > largestGraph = null;
//			for ( final Set< Tile< ? > > graph : graphs )
//				if ( largestGraph == null || largestGraph.size() < graph.size() )
//					largestGraph = graph;
//
//			interestingTiles = new ArrayList< AbstractAffineTile2D< ? > >();
//			for ( final Tile< ? > t : largestGraph )
//				interestingTiles.add( ( AbstractAffineTile2D< ? > )t );
//
//			if ( hideDisconnectedTilesIn )
//				for ( final AbstractAffineTile2D< ? > t : tiles )
//					if ( !interestingTiles.contains( t ) )
//						t.getPatch().setVisible( false );
//			if ( deleteDisconnectedTilesIn )
//				for ( final AbstractAffineTile2D< ? > t : tiles )
//					if ( !interestingTiles.contains( t ) )
//						t.getPatch().remove( false );
//		}
//		else
//			interestingTiles = tiles;
		
		final Collection< Tile< ? > > tiles = tilesMap.values();
		final TileConfiguration tc = new TileConfiguration();
		for ( final Tile< ? > t : tiles )
			if ( t.getConnectedTiles().size() > 0 )
				tc.addTile( t );

		for ( final Tile< ? > t : fixedTiles )
			tc.fixTile( t );

		try
		{
			if ( params.filterOutliers )
				tc.optimizeAndFilter( params.maxEpsilon, params.maxIterations, params.maxPlateauwidth, params.meanFactor );
			else
				tc.optimize( params.maxEpsilon, params.maxIterations, params.maxPlateauwidth );
		}
		catch ( final Exception e )
		{
			System.err.println( "Error optimizing:" );
			e.printStackTrace( System.err );
		}
		
		ArrayList< TileSpec > out_tiles = new ArrayList< TileSpec >();
				
		// Export new transforms, TODO: append to existing tilespec files
		for(Entry<String, Tile< ? > > entry : tilesMap.entrySet()) {
		    String tile_url = entry.getKey();
		    Tile< ? > tile_value = entry.getValue();
		    
		    TileSpec ts = new TileSpec();
		    ts.imageUrl = tile_url;
		    
		    @SuppressWarnings("rawtypes")
			Model genericModel = tile_value.getModel();
		    
		    Transform addedTransform = new Transform();
		    addedTransform.className = genericModel.getClass().getCanonicalName();
		    
			switch ( params.modelIndex )
			{
			case 0:
				addedTransform.dataString = ((TranslationModel2D) genericModel).toDataString();
			case 1:
				addedTransform.dataString = ((RigidModel2D) genericModel).toDataString();
			case 2:
				addedTransform.dataString = ((SimilarityModel2D) genericModel).toDataString();
			case 3:
				addedTransform.dataString = ((AffineModel2D) genericModel).toDataString();
			case 4:
				addedTransform.dataString = ((HomographyModel2D) genericModel).toDataString();
			default:
				addedTransform.dataString = genericModel.toString();
			}		    
		    
		    ts.transforms = new Transform[]{addedTransform};
		    
		    out_tiles.add(ts);
		}
		
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
	}
	
}

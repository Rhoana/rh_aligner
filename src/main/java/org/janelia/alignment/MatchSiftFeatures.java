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

import java.io.FileReader;
import java.io.IOException;
import java.io.Writer;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import org.janelia.alignment.FeatureSpec.ImageAndFeatures;

import mpicbg.models.PointMatch;
import mpicbg.imagefeatures.Feature;
import mpicbg.ij.FeatureTransform;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

/**
 *
 * @author Seymour Knowles-Barley
 */
public class MatchSiftFeatures
{
	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

        @Parameter( names = "--featurefile", description = "Feature file", required = true )
        private String featurefile;

        @Parameter( names = "--targetPath", description = "Path for the output correspondences", required = true )
        public String targetPath;

        @Parameter( names = "--indices", description = "Pair of indices within feature file, comma separated (each pair is separated by a colon)", required = true )
        public List<String> indices = new ArrayList<String>();

        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

        @Parameter( names = "--rod", description = "ROD", required = false )
        public float rod = 0.5f;
	}

	private MatchSiftFeatures() {}

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

		/* open featurespec */
		final FeatureSpec[] featureSpecs;
		try
		{
			final Gson gson = new Gson();
			featureSpecs = gson.fromJson( new FileReader( params.featurefile.replace("file://", "").replace("file:/", "") ), FeatureSpec[].class );
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

		List< CorrespondenceSpec > corr_data = new ArrayList< CorrespondenceSpec >();

		for (String idx_pair : params.indices) {
			String[] vals = idx_pair.split(":");
			if (vals.length != 2)
				throw new IllegalArgumentException("Index pair not in correct format:" + idx_pair);
			int idx1 = Integer.parseInt(vals[0]);
			int idx2 = Integer.parseInt(vals[1]);
			
			final ImageAndFeatures iaf1 = featureSpecs[idx1].getMipmapImageAndFeatures( mipmapLevel );
			final ImageAndFeatures iaf2 = featureSpecs[idx2].getMipmapImageAndFeatures( mipmapLevel );

			final List< Feature > fs1 = iaf1.featureList;
			final List< Feature > fs2 = iaf2.featureList;

			final List< PointMatch > candidates = new ArrayList< PointMatch >();
			FeatureTransform.matchFeatures( fs1, fs2, candidates, params.rod );

			corr_data.add(new CorrespondenceSpec(mipmapLevel,
					iaf1.imageUrl,
					iaf2.imageUrl,
					candidates));
		}

		if (corr_data.size() > 0) {
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
}

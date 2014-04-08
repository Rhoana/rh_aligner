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

import ij.IJ;
import ij.ImagePlus;
import ij.io.Opener;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.FileReader;
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

import javax.imageio.ImageIO;

import org.janelia.alignment.OptimizeMontageTransform.Params;

import mpicbg.models.AbstractModel;
import mpicbg.models.AffineModel2D;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;
import mpicbg.models.CoordinateTransformMesh;
import mpicbg.models.ErrorStatistic;
import mpicbg.models.HomographyModel2D;
import mpicbg.models.InvertibleCoordinateTransform;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.RigidModel2D;
import mpicbg.models.SimilarityModel2D;
import mpicbg.models.SpringMesh;
import mpicbg.models.TransformMesh;
import mpicbg.models.TranslationModel2D;
import mpicbg.models.Vertex;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks;
import mpicbg.trakem2.transform.TransformMeshMappingWithMasks.ImageProcessorWithMasks;
import mpicbg.imagefeatures.Feature;
import mpicbg.imagefeatures.FloatArray2DSIFT;
import mpicbg.ij.FeatureTransform;
import mpicbg.ij.SIFT;
import mpicbg.ij.blockmatching.BlockMatching;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonSyntaxException;

/** 
 * @author Seymour Knowles-Barley
 */
public class OptimizeSeriesTransform
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
	
	private OptimizeSeriesTransform() {}
	
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

		System.out.printl( "matching " + layerNameB + " -> " + layerNameA + "..." );

		ArrayList< PointMatch > candidates = null;
		if ( !param.ppm.clearCache )
			candidates = mpicbg.trakem2.align.Util.deserializePointMatches(
					layerB.getProject(), param.ppm, "layer", layerB.getId(), layerA.getId() );

		if ( null == candidates )
		{
			final ArrayList< Feature > fs1 = mpicbg.trakem2.align.Util.deserializeFeatures(
					layerA.getProject(), param.ppm.sift, "layer", layerA.getId() );
			final ArrayList< Feature > fs2 = mpicbg.trakem2.align.Util.deserializeFeatures(
					layerB.getProject(), param.ppm.sift, "layer", layerB.getId() );
			candidates = new ArrayList< PointMatch >( FloatArray2DSIFT.createMatches( fs2, fs1, param.ppm.rod ) );

			/* scale the candidates */
			for ( final PointMatch pm : candidates )
			{
				final Point p1 = pm.getP1();
				final Point p2 = pm.getP2();
				final float[] l1 = p1.getL();
				final float[] w1 = p1.getW();
				final float[] l2 = p2.getL();
				final float[] w2 = p2.getW();

				l1[ 0 ] *= pointMatchScale;
				l1[ 1 ] *= pointMatchScale;
				w1[ 0 ] *= pointMatchScale;
				w1[ 1 ] *= pointMatchScale;
				l2[ 0 ] *= pointMatchScale;
				l2[ 1 ] *= pointMatchScale;
				w2[ 0 ] *= pointMatchScale;
				w2[ 1 ] *= pointMatchScale;

			}

			if ( !mpicbg.trakem2.align.Util.serializePointMatches(
					layerB.getProject(), param.ppm, "layer", layerB.getId(), layerA.getId(), candidates ) )
			Utils.log( "Could not store point match candidates for layers " + layerNameB + " and " + layerNameA + "." );
		}


		final ArrayList< PointMatch > inliers = new ArrayList< PointMatch >();

		boolean again = false;
		int nHypotheses = 0;
		try
		{
			do
			{
				again = false;
				final ArrayList< PointMatch > inliers2 = new ArrayList< PointMatch >();
				final boolean modelFound = model.filterRansac(
							candidates,
							inliers2,
							1000,
							param.maxEpsilon,
							param.minInlierRatio,
							param.minNumInliers,
							3 );
				if ( modelFound )
				{
					candidates.removeAll( inliers2 );

					if ( param.rejectIdentity )
					{
						final ArrayList< Point > points = new ArrayList< Point >();
						PointMatch.sourcePoints( inliers2, points );
						if ( Transforms.isIdentity( model, points, param.identityTolerance ) )
						{
							IJ.log( "Identity transform for " + inliers2.size() + " matches rejected." );
							again = true;
						}
						else
						{
							++nHypotheses;
							inliers.addAll( inliers2 );
							again = param.multipleHypotheses;
						}
					}
					else
					{
						++nHypotheses;
						inliers.addAll( inliers2 );
						again = param.multipleHypotheses;
					}
				}
			}
			while ( again );
		}
		catch ( final NotEnoughDataPointsException e ) {}

		if ( nHypotheses > 0 && param.multipleHypotheses )
		{
			try
			{
					model.fit( inliers );
					PointMatch.apply( inliers, model );
			}
			catch ( final NotEnoughDataPointsException e ) {}
			catch ( final IllDefinedDataPointsException e )
			{
				nHypotheses = 0;
			}
		}

		if ( nHypotheses > 0 )
		{								
			Utils.log( layerNameB + " -> " + layerNameA + ": " + inliers.size() + " corresponding features with an average displacement of " + ( PointMatch.meanDistance( inliers ) ) + "px identified." );
			Utils.log( "Estimated transformation model: " + model + ( param.multipleHypotheses ? ( " from " + nHypotheses + " hypotheses" ) : "" ) );
			models.set( ti, new Triple< Integer, Integer, Collection< PointMatch > >( sliceA, sliceB, inliers ) );
		}
		else
		{
			Utils.log( layerNameB + " -> " + layerNameA + ": no correspondences found." );
			return;
		}
		
	}
	
}

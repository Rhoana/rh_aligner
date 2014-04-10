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

import java.util.List;
import java.util.TreeMap;

import mpicbg.imagefeatures.Feature;

/**
 * 
 *
 * @author Seymour Knowles-Barley
 */
public class FeatureSpec
{

	public class ImageAndFeatures
	{
		public String imageUrl;
		public List< Feature > featureList;
		
		public ImageAndFeatures(String url, List< Feature > flist)
		{
			imageUrl = url;
			featureList = flist;
		}

	}
	
	final private TreeMap< String, ImageAndFeatures > mipmapLevels = new TreeMap< String, ImageAndFeatures >();

	public int mipmapLevel;
	
	public FeatureSpec(String mipmapLevel, String url, List< Feature > flist)
	{
		ImageAndFeatures iaf = new ImageAndFeatures(url, flist);
		mipmapLevels.put(mipmapLevel, iaf);
	}
		
	final public ImageAndFeatures getMipmapImageAndFeatures(int mipmapLevel)
	{
		return mipmapLevels.get("" + mipmapLevel);
	}

}

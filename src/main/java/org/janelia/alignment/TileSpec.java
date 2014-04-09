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

import mpicbg.models.CoordinateTransform;
import mpicbg.models.CoordinateTransformList;

/**
 * 
 *
 * @author Stephan Saalfeld <saalfelds@janelia.hhmi.org>, Seymour Knowles-Barley
 */
public class TileSpec
{
	public String imageUrl;
	public String maskUrl;
	public double minIntensity = 0;
	public double maxIntensity = 255;
	// width and height of base tile image
	public int width = 0;
	public int height = 0;
	// bounding box after transformations are applied [left, right, top, bottom] possibly with extra entries for [front, back, etc.]
	public float[] bbox = null;
	public Transform[] transforms = null;
	
	final public CoordinateTransformList< CoordinateTransform > createTransformList()
	{
		final CoordinateTransformList< CoordinateTransform > ctl = new CoordinateTransformList< CoordinateTransform >();
		
		if ( transforms != null )
			for ( final Transform t : transforms )
				ctl. add( t.createTransform() );
		
		return ctl;
	}
}

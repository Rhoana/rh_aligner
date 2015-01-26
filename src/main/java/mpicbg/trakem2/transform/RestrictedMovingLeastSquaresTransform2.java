package mpicbg.trakem2.transform;


import java.util.List;

import mpicbg.models.AffineModel2D;
import mpicbg.models.AffineModel3D;
import mpicbg.models.IllDefinedDataPointsException;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.PointMatch;
import mpicbg.models.RigidModel2D;
import mpicbg.models.SimilarityModel2D;
import mpicbg.models.TranslationModel2D;

/**
 * The same as moving least squares transform, but restricted to a neighboring vertices
 * when applying a transformation
 * 
 * @author adisuis
 *
 */
public class RestrictedMovingLeastSquaresTransform2 extends mpicbg.models.RestrictedMovingLeastSquaresTransform2 implements CoordinateTransform
{

	@Override
	public void init( final String data ) throws NumberFormatException
	{
		final String[] fields = data.split( "\\s+" );
		if ( fields.length > 3 )
		{
			final int dim = Integer.parseInt( fields[ 1 ] );
			
			if ( ( fields.length - 4 ) % ( 2 * dim + 1 ) == 0 )
			{
				final int l = ( fields.length - 4 ) / ( 2 * dim + 1 );
				
				if ( dim == 2 )
				{
					if ( fields[ 0 ].equals( "translation" ) ) model = new TranslationModel2D();
					else if ( fields[ 0 ].equals( "rigid" ) ) model = new RigidModel2D();
					else if ( fields[ 0 ].equals( "similarity" ) ) model = new SimilarityModel2D();
					else if ( fields[ 0 ].equals( "affine" ) ) model = new AffineModel2D();
					else throw new NumberFormatException( "Inappropriate parameters for " + this.getClass().getCanonicalName() );
				}
				else if ( dim == 3 )
				{
					if ( fields[ 0 ].equals( "affine" ) ) model = new AffineModel3D();
					else throw new NumberFormatException( "Inappropriate parameters for " + this.getClass().getCanonicalName() );
				}
				else throw new NumberFormatException( "Inappropriate parameters for " + this.getClass().getCanonicalName() );
				
				alpha = Float.parseFloat( fields[ 2 ] );

				neighborsRadius = Float.parseFloat( fields[ 3 ] );
				
				p = new float[ dim ][ l ];
				q = new float[ dim ][ l ];
				w = new float[ l ];
				
				int i = 3, j = 0;
				while ( i < fields.length - 1 )
				{
					for ( int d = 0; d < dim; ++d )
						p[ d ][ j ] = Float.parseFloat( fields[ ++i ] );
					for ( int d = 0; d < dim; ++d )
						q[ d ][ j ] = Float.parseFloat( fields[ ++i ] );
					w[ j ] = Float.parseFloat( fields[ ++i ] );
					++j;
				}
			}
			else throw new NumberFormatException( "Inappropriate parameters for " + this.getClass().getCanonicalName() );
		}
		else throw new NumberFormatException( "Inappropriate parameters for " + this.getClass().getCanonicalName() );

	}

	@Override
	public String toDataString()
	{
		final StringBuilder data = new StringBuilder();
		toDataString( data );
		return data.toString();
	}

	private void toDataString( final StringBuilder data )
	{
		if ( AffineModel2D.class.isInstance( model ) )
			data.append( "affine 2" );
		else if ( TranslationModel2D.class.isInstance( model ) )
			data.append( "translation 2" );
		else if ( RigidModel2D.class.isInstance( model ) )
			data.append( "rigid 2" );
		else if ( SimilarityModel2D.class.isInstance( model ) )
			data.append( "similarity 2" );
		else if ( AffineModel3D.class.isInstance( model ) )
			data.append( "affine 3" );
		else
			data.append( "unknown" );

		data.append(' ').append(alpha);

		data.append(' ').append(neighborsRadius);

		final int n = p.length;
		final int l = p[ 0 ].length;
		for ( int i = 0; i < l; ++i )
		{
			for ( int d = 0; d < n; ++d )
				data.append(' ').append( p[ d ][ i ] );
			for ( int d = 0; d < n; ++d )
				data.append(' ').append( q[ d ][ i ] );
			data.append(' ').append( w[ i ] );
		}
	}

	@Override
	public String toXML( final String indent )
	{
		final StringBuilder xml = new StringBuilder( 80000 );
		xml.append( indent )
		   .append( "<ict_transform class=\"" )
		   .append( this.getClass().getCanonicalName() )
		   .append( "\" data=\"" );
		toDataString( xml );
		return xml.append( "\"/>" ).toString();
	}

	@Override
	public RestrictedMovingLeastSquaresTransform2 copy()
	{
		final RestrictedMovingLeastSquaresTransform2 t = new RestrictedMovingLeastSquaresTransform2();
		t.model = this.model.copy();
		t.alpha = this.alpha;
		// Copy p, q, w
		t.p = new float[this.p.length][this.p[0].length];
		//
		for (int i=0; i<this.p.length; ++i)
			for (int k=0; k<this.p[0].length; ++k)
				t.p[i][k] = this.p[i][k];
		//
		t.q = new float[this.q.length][this.q[0].length];
		//
		for (int i=0; i<this.q.length; ++i)
			for (int k=0; k<this.q[0].length; ++k)
				t.q[i][k] = this.q[i][k];
		//
		t.w = new float[this.w.length];
		//
		for (int i=0; i<this.w.length; ++i)
			t.w[i] = this.w[i];
		
		t.neighborsRadius = neighborsRadius;
		return t;
	}

	@Override
	public RestrictedMovingLeastSquaresTransform2 boundToBoundingBox(
			final float[] boundingBox )
	{
		List< PointMatch > filteredMatches = filterMatchesWithHalo( boundingBox );
		RestrictedMovingLeastSquaresTransform2 boundedRMLT = new RestrictedMovingLeastSquaresTransform2();
		boundedRMLT.setModel( this.model );
		boundedRMLT.setAlpha( this.alpha );
		try {
			boundedRMLT.setMatches( filteredMatches );
		} catch (NotEnoughDataPointsException e) {
			// Should not happen
			throw new RuntimeException( e );
		} catch (IllDefinedDataPointsException e) {
			// Should not happen
			throw new RuntimeException( e );
		}		
		boundedRMLT.setRadius( this.neighborsRadius );

		return boundedRMLT;
	}

}

package org.janelia.alignment;

import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;

public class TileSpecUtils {

	public static TileSpec[] readTileSpecFile( String tsUrl )
	{
		/* open tilespec */
		final URL url;
		final TileSpec[] tileSpecs;
		try
		{
			final Gson gson = new Gson();
			url = new URL( tsUrl );
			tileSpecs = gson.fromJson( new InputStreamReader( url.openStream() ), TileSpec[].class );
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
		return tileSpecs;
	}

	public static TileSpec[] readTileSpecFile( String[] tsUrls )
	{
		final List< TileSpec > allTileSpecs = new ArrayList<TileSpec>();

		/* open tilespec */
		final Gson gson = new Gson();
		try
		{
			for ( String tsUrl : tsUrls )
			{
				final URL url = new URL( tsUrl );
				final TileSpec[] tileSpecs = gson.fromJson( new InputStreamReader( url.openStream() ), TileSpec[].class );
				/* add all tile specs to the list */
				for ( TileSpec ts : tileSpecs )
					allTileSpecs.add( ts );
			}
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
		TileSpec[] tileSpecsArr = new TileSpec[ allTileSpecs.size() ];
		return allTileSpecs.toArray( tileSpecsArr );
	}

	
}

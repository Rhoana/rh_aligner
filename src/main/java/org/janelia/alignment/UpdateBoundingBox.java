package org.janelia.alignment;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

/**
 * Updates the bounding box of each tilespec of the given tilespec files,
 * and outputs the tile spec to a file with a given suffix
 * 
 */
public class UpdateBoundingBox {

	@Parameters
	static private class Params
	{
		@Parameter( names = "--help", description = "Display this note", help = true )
        private final boolean help = false;

		@Parameter(description = "Json files to update bounding box for")
		private List<String> files = new ArrayList<String>();
		        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

        @Parameter( names = "--suffix", description = "The suffix to add to the out file", required = false )
        public String filesSuffix = "_bbox";
        
	}

	private UpdateBoundingBox() { }
	
	final static Params parseParams( final String[] args )
	{
		final Params params = new Params();
		try
        {
			final JCommander jc = new JCommander( params, args );
        	if ( params.help )
            {
        		jc.usage();
                return null;
            }
        }
        catch ( final Exception e )
        {
        	e.printStackTrace();
            final JCommander jc = new JCommander( params );
        	jc.setProgramName( "java [-options] -cp render.jar + " + Render.class.getCanonicalName() );
        	jc.usage(); 
        	return null;
        }
		
		return params;
	}
	
	private static final void updateFileBoundingBox( final String fileName, final String filesSuffix, final int threadsNum )
	{
		final TileSpecsImage tsImage = TileSpecsImage.createImageFromFile( fileName );
		// Set a single thread per image 
		tsImage.setThreadsNum( threadsNum );
		
		tsImage.getBoundingBox( true );
		
		// Save the image
		String outFileName = fileName.replace( ".json", filesSuffix + ".json" );
		outFileName = outFileName.replace( "file://", "" );
		tsImage.saveTileSpecs( outFileName );
	}
	
	public static void main( final String[] args )
	{		
		final Params params = parseParams( args );
		
		if ( params == null )
			return;
		

		if ( params.numThreads <= params.files.size() )
		{
			// Each thread updates the bbox of a single json file
			final ExecutorService threadPool = Executors.newFixedThreadPool( params.numThreads );
			final List< Future< ? > > futures = new ArrayList< Future< ? >>();

			for ( final String fileName : params.files )
			{
				final Future< ? > future = threadPool.submit( new Runnable() {
					
					@Override
					public void run() {
						updateFileBoundingBox( fileName, params.filesSuffix, 1 );
					}
				} );
				futures.add( future );
			}
			
			try {
				for ( Future< ? > future : futures ) {
					future.get();
				}
			} catch ( InterruptedException e ) {
				e.printStackTrace();
				throw new RuntimeException( e );
			} catch ( ExecutionException e ) {
				e.printStackTrace();
				throw new RuntimeException( e );
			}
			threadPool.shutdown();
		}
		else
		{
			// Each update of json file's bbox is done using multiple threads
			for ( final String fileName : params.files )
			{
				updateFileBoundingBox( fileName, params.filesSuffix, params.numThreads );
			}
		}
	}
}

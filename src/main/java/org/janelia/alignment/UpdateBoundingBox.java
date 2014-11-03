package org.janelia.alignment;

import java.io.File;
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
 * and outputs the tile spec to a file with the same name in a given output directory
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
		        
        @Parameter( names = "--targetDir", description = "The directory where the output json files will be saved (SectionNNN.json)", required = true )
        public String targetDir;
        
        @Parameter( names = "--threads", description = "Number of threads to be used", required = false )
        public int numThreads = Runtime.getRuntime().availableProcessors();

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
	
	private static final void updateFileBoundingBox( final String fileName, final String targetDir, final int threadsNum )
	{
		final TileSpecsImage tsImage = TileSpecsImage.createImageFromFile( fileName );
		// Set a single thread per image 
		tsImage.setThreadsNum( threadsNum );
		
		tsImage.getBoundingBox( true );
		
		final String outFileName = targetDir + fileName.substring( fileName.lastIndexOf( '/' ) );
		
		// Save the image
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
						updateFileBoundingBox( fileName, params.targetDir, 1 );
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
				updateFileBoundingBox( fileName, params.targetDir, params.numThreads );
			}
		}
	}
}

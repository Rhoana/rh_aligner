package org.janelia.alignment;

import java.util.concurrent.ExecutionException;

import mpicbg.models.ErrorStatistic;
import mpicbg.models.IllDefinedDataPointsException;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.TileUtil;

public class TileConfiguration extends mpicbg.models.TileConfiguration {

	protected int threadsNum;
	
	public TileConfiguration() {
		this.threadsNum = Runtime.getRuntime().availableProcessors();
	}
	
	public void setThreadsNum( int threadsNum ) {
		this.threadsNum = threadsNum;
	}
	
	public void optimizeSilentlyConcurrent(
			final ErrorStatistic observer,
			final float maxAllowedError,
			final int maxIterations,
			final int maxPlateauwidth ) throws NotEnoughDataPointsException, IllDefinedDataPointsException, InterruptedException, ExecutionException 
	{
		TileUtil.optimizeConcurrently( observer, maxAllowedError, maxIterations, maxPlateauwidth,
				this, tiles, fixedTiles, threadsNum );
	}

}

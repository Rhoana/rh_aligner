package org.janelia.alignment;

/**
 * A Work (indices) distributer between different workers, in order to divide the items as equally as
 * possible between the workers.
 * @author adisuis
 *
 */
public class Distributer {

	private int div;
	private int mod;
	private int curWorker;
	private int curStart;
	private int curEnd;
	
	public Distributer( int items, int workers ) {
		this.curWorker = 0;
		this.div = items / workers;
		this.mod = items % workers;
		this.curStart = 0;
		this.curEnd = this.div;
		if ( this.mod > 0 ) // The first worker receives its portion and also 1 of the remainders if there are any
			this.curEnd += 1;
	}
	
	public int getStart() {
		return curStart;
	}

	public int getEnd() {
		return curEnd;
	}

	public void next() {
		curStart = curEnd;
		curWorker++;
		curEnd = curEnd + div;
		if ( curWorker < mod )
			curEnd = curEnd + 1;
	}
}

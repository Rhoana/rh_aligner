package org.janelia.alignment;

import java.util.NoSuchElementException;

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
	private int items;
	
	public Distributer( int items, int workers ) {
		this.curWorker = 0;
		this.items = items;
		this.div = items / workers;
		this.mod = items % workers;
		this.curStart = 0;
		this.curEnd = 0;
	}
	
	public int getStart() {
		return curStart;
	}

	public int getEnd() {
		return curEnd;
	}
	
	public boolean hasNext() {
		return curEnd < items;
	}

	public void next() {
		if (! hasNext() )
			throw new NoSuchElementException();
		
		curStart = curEnd;
		curEnd = curEnd + div;
		if ( curWorker < mod ) // The first workers receive their portion and also 1 of the remainders if there are any
			curEnd = curEnd + 1;
		curWorker++;
	}
}

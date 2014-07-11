package org.janelia.alignment;

import mpicbg.models.CoordinateTransform;



public class ModelSpec {
	
	public String url1; // the first layer json tile spec url
	public String url2; // the second layer json tile spec url
	public Transform model = null; // the coordinate transformation model


	public ModelSpec( String url1, String url2, Transform model ) {
		this.url1 = url1;
		this.url2 = url2;
		this.model = model;
	}
	
	public CoordinateTransform createModel() {
		CoordinateTransform result = null;
		if ( model != null ) {
			result = model.createTransform();
		}
		return result;
	}
	
}

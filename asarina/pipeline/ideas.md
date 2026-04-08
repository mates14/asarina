 375 -        # 6. Refined photometry                                                                                                                          
 376 -        ecsv = pipeline.photometry(fits_file, temp_dir)                                                                                                  
 377 -        if ecsv is None:                                                                                                                                 
 378 -            sys.exit(1)                                                                                                                                  
 379 -                                                                                                                                                         
 380 -        # 7. Update archived raw with the photometry-refined WCS (needs root).                                                                           
 381 -        #    If dft.fits doesn't exist, pyrt-dophot didn't refine the solution                                                                           
 382 -        #    and the archive already has the field-solve WCS from step 4 — skip.                                                                         
 383 -        # DISABLE until it is at least somewhat fixed                                                                                                    
 384 -        # dft = temp_dir / fits_file.replace('df.fits', 'dft.fits')                                                                                      
 385 -        # if ctime is not None:                                                                                                                          
 386 -        #    if dft.exists():                                                                                                                            
 387 -        #        _update_archive(raw_path, dft, ctime, chip_id, ccd_name,                                                                                
 388 -        #                        ecsv_path=ecsv_path)                                                                                                    
 389 -        #    else:                                                                                                                                       
 390 -        #        logger.debug("dft.fits not found, archive WCS already up to date")       

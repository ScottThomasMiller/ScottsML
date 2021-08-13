//
//  EEG.swift
//  BCILab
//
//  Created by Scott Miller on 8/12/21.
//

import Foundation

struct EEG {
    let eegFilename = "BrainLabEEG.csv"
        
    func streamToFile() {
        print("resolving EEG stream")
        let eegStreamInfo = resolve(property: "type", value: "EEG")
        let inlet = Inlet(streamInfo: eegStreamInfo[0])
        var buffer = [Float32]()
        var timestamp: Double = 0.0
        var timeCorrection: Double = 0.0
        
        do {
            print("opening inlet")
            try inlet.openStream(timeout: 60) }
        catch {
            print("Cannot open stream.  Error: \(error)")
            return
        }
        
        do {
            print("inlet is open. getting time correction")
            try timeCorrection = inlet.timeCorrection() }
        catch {
            print("Cannot get time correction.  Error: \(error)")
            return
        }
        print("time correction = \(timeCorrection)")
        
        while true {
            do {
                try (timestamp, buffer) = inlet.pullSample() }
            catch {
                print("Sample error: \(error)")
            }
            //print("timestamp: \(timestamp) buffer: \(buffer)")
        }
    }
    
}

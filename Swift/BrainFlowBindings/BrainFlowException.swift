//
//  BrainFlowException.swift
//  A wrapper for BrainFlow's brainflow_exception.h
//
//  Created by Scott Miller on 8/23/21.
//

import Foundation

class BrainFlowException: Error {
    var message: String
    var errorCode: Int32
    
    init(_ errorMessage: String, _ code: Int32) {
        message = errorMessage
        errorCode = code
    }
}

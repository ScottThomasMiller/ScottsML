//
//  BoardShim.swift
//  A wrapper mirroring BrainFlow's board_shim.cpp
//
//  Created by Scott Miller for Aeris Rising, LLC on 8/23/21.
//

import Foundation

struct BoardShim {
    let boardId: BoardIds
    let params: BrainFlowInputParams
    let jsonParams: String
    
    init (_ boardId: BoardIds, _ params: BrainFlowInputParams) {
        self.boardId = boardId
        self.params = params
        self.jsonParams = params.json()
    }
    
    func prepareSession() throws {
        let result = prepare_session(boardId.rawValue, jsonParams)
        let exitCode = BrainFlowExitCodes(rawValue: result)
        if exitCode != BrainFlowExitCodes.STATUS_OK {
            throw BrainFlowException("failed to prepare session", result)
        }
    }
    
    func isPrepared () throws -> Bool  {
        var intPrepared: Int32 = 0
        let result = is_prepared (&intPrepared, boardId.rawValue, jsonParams)
        let exitCode = BrainFlowExitCodes(rawValue: result)
        if exitCode != BrainFlowExitCodes.STATUS_OK {
            throw BrainFlowException ("failed to check session", result)
        }
        guard let boolPrepared = Bool(exactly: NSNumber(value: intPrepared)) else {
            throw BrainFlowException ("is_prepared returned non-boolean", intPrepared)
        }
        return boolPrepared
    }

    func releaseSession () throws {
        let result = release_session (boardId.rawValue, jsonParams)
        let exitCode = BrainFlowExitCodes(rawValue: result)
        if exitCode != BrainFlowExitCodes.STATUS_OK {
            throw BrainFlowException ("failed to release session", result);
        }
    }

}

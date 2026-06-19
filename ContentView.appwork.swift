import SwiftUI
import UniformTypeIdentifiers
import PDFKit
import UIKit

// MARK: - Configuration

private enum AccentTheme: String, CaseIterable {
    case classic, orange, mint, skyBlue, indigo, pink

    var color: Color {
        switch self {
        case .classic: return .blue
        case .orange:  return .orange
        case .mint:    return Color(red: 0.20, green: 0.78, blue: 0.60)
        case .skyBlue: return Color(red: 0.30, green: 0.68, blue: 0.95)
        case .indigo:  return .indigo
        case .pink:    return Color(red: 0.96, green: 0.45, blue: 0.60)
        }
    }

    /// How much accent color to mix into card backgrounds (0 = pure system gray)
    var backgroundTint: Double {
        switch self {
        case .classic: return 0.0
        default:       return 0.06
        }
    }

    var label: String {
        switch self {
        case .classic: return "Default"
        case .orange:  return "Orange"
        case .mint:    return "Mint"
        case .skyBlue: return "Sky Blue"
        case .indigo:  return "Indigo"
        case .pink:    return "Pink"
        }
    }
}

private enum BackendConfig {
    static let baseURL = URL(string: "https://omr-trigger-777135743132.us-central1.run.app")!
    static let pollSeconds: UInt64 = 2
    static let maxPollAttempts = 180
    static let artifactRetryAttempts = 8
    static let artifactRetryDelaySeconds: UInt64 = 2
}

private enum FrontendDebugConfig {
    static let debugSafeJobIDs = true
    static let disableAutoResume = false
    static let overlayLoggingEnabled = true
    static let overlayLogMaxRowsPerPage = 5
}

// MARK: - State Machine
// Transitions:
//   idle → uploading → dispatching → processing → ready
//                                               ↘ failed
//   Any phase → idle  (via clearSession)

private enum AppPhase: String {
    case idle
    case uploading
    case dispatching
    case processing
    case ready
    case failed
}

// MARK: - Data Models

private struct UploadResponse: Decodable {
    let pdf_gcs_uri: String
}

private struct CreateJobRequest: Encodable {
    let pdf_gcs_uri: String
    let job_id: String
}

private struct CreateJobResponse: Decodable {
    let job_id: String
    let run_id: Int?
    let artifacts_http: ArtifactHTTP?
}

private struct JobStatusResponse: Decodable {
    let job_id: String
    let run_id: Int?
    let status: String?
    let artifacts_http: ArtifactHTTP?
}

private struct ArtifactHTTP: Decodable {
    let audiveris_out_pdf: String?
    let audiveris_out_corrected_pdf: String?
    let mapping_summary: String?
}

private struct JobStateResponse: Decodable {
    var editable_state: EditableState
    let ai_suggestions: AISuggestionsState?
    let ai_suggest_run: AISuggestRunState?
    let artifacts_http: ArtifactHTTP?
}

private struct BackendErrorPayload: Decodable {
    let code: String?
    let message: String?
    let detail: String?
    let provider_status: Int?
}

private struct AISuggestRunState: Decodable {
    let status: String?
    let started_at_utc: String?
    let updated_at_utc: String?
    let completed_at_utc: String?
    let failed_at_utc: String?
    let systems_total: Int?
    let systems_completed: Int?
    let next_system_index: Int?
    let source_run_id: Int?
    let source_state_version: String?
    let last_error: BackendErrorPayload?
}

private struct AISuggestionsState: Decodable {
    let version: String?
    let generated_at_utc: String?
    let provider: String?
    let model: String?
    let source_run_id: Int?
    let source_state_version: String?
    let by_measure_id: [String: AISuggestionEntry]?
    let warnings: [AISuggestionWarning]?
    let summary: AISuggestionSummary?
}

private struct AISuggestionEntry: Decodable {
    let label: String
    let rest_count: Int?
    let confidence: String?
    let maybe_label: String?
    let maybe_rest_count: Int?
    let system_id: String?
    let order_index_in_system: Int?
    let is_first_measure_of_score: Bool?
}

private struct AISuggestionSummary: Decodable {
    let systems_processed: Int?
    let measures_seen: Int?
    let suggestions_kept: Int?
    let normal_measures_omitted: Int?
}

private struct AISuggestionWarning: Decodable {
    let type: String?
    let system_id: String?
    let system_index: Int?
    let message: String?
}

private struct AISuggestResponse: Decodable {
    let job_id: String?
    let run_id: Int?
    let status: String?
    let ai_suggestions: AISuggestionsState?
    let ai_suggest_run: AISuggestRunState?
    let error: BackendErrorPayload?
}

private struct AIDismissResponse: Decodable {
    let job_id: String?
    let run_id: Int?
    let status: String?
    let dismissed_measure_id: String?
    let ai_suggestions: AISuggestionsState?
}

private struct EditableState: Decodable {
    let labels_mode: String?
    let systems: [SystemState]
    let measures: [MeasureState]?
    var staff_boxes: [StaffBoxState]?
    let auto_rows: [AutoRowState]?
    let measure_number_overrides: [String: Int]?
    let rest_measures: [String: Int]?
    let pickup_measures: [String: Bool]?
    let endings: [String: String]?
    let manual_rows: [ManualRowState]?
}

private struct EditColorStyle: Identifiable {
    let id: String
    let title: String
    let fill: UIColor
    let stroke: UIColor

    var swiftFill: Color { Color(uiColor: fill) }
    var swiftStroke: Color { Color(uiColor: stroke) }
}

private enum EditColorPalette {
    static let normal = EditColorStyle(
        id: "normal",
        title: "Normal: no edit",
        fill: UIColor(red: 0.83, green: 0.96, blue: 0.86, alpha: 0.55),
        stroke: UIColor(red: 0.20, green: 0.63, blue: 0.31, alpha: 0.92)
    )

    static let measureNumber = EditColorStyle(
        id: "measure_number",
        title: "Set Measure #",
        fill: UIColor(red: 0.79, green: 0.89, blue: 1.00, alpha: 0.60),
        stroke: UIColor(red: 0.14, green: 0.42, blue: 0.88, alpha: 0.95)
    )

    static let rest = EditColorStyle(
        id: "rest",
        title: "Rest",
        fill: UIColor(red: 1.00, green: 0.90, blue: 0.78, alpha: 0.60),
        stroke: UIColor(red: 0.94, green: 0.55, blue: 0.19, alpha: 0.94)
    )

    static let pickup = EditColorStyle(
        id: "pickup",
        title: "Pickup",
        fill: UIColor(red: 1.00, green: 0.95, blue: 0.70, alpha: 0.62),
        stroke: UIColor(red: 0.72, green: 0.54, blue: 0.05, alpha: 0.94)
    )

    static let aiSuggestion = EditColorStyle(
        id: "ai_suggestion",
        title: "AI Suggestion",
        fill: UIColor(red: 0.74, green: 0.91, blue: 1.00, alpha: 0.64),
        stroke: UIColor(red: 0.00, green: 0.50, blue: 0.80, alpha: 0.96)
    )

    static let ending1 = EditColorStyle(
        id: "ending_1",
        title: "Ending 1",
        fill: UIColor(red: 1.00, green: 0.84, blue: 0.89, alpha: 0.62),
        stroke: UIColor(red: 0.84, green: 0.30, blue: 0.50, alpha: 0.94)
    )

    static let ending2 = EditColorStyle(
        id: "ending_2",
        title: "Ending 2",
        fill: UIColor(red: 0.88, green: 0.84, blue: 1.00, alpha: 0.62),
        stroke: UIColor(red: 0.45, green: 0.29, blue: 0.83, alpha: 0.95)
    )

    static let excluded = EditColorStyle(
        id: "excluded",
        title: "Excluded",
        fill: UIColor(red: 0.84, green: 0.86, blue: 0.89, alpha: 0.68),
        stroke: UIColor(red: 0.32, green: 0.36, blue: 0.42, alpha: 0.96)
    )

    static let legendItems: [EditColorStyle] = [
        normal,
        excluded,
        measureNumber,
        rest,
        pickup,
        aiSuggestion,
        ending1,
        ending2
    ]
}

private enum LabelsMode: String, CaseIterable, Identifiable {
    case systemOnly = "system_only"
    case allMeasures = "all_measures"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .systemOnly: return "Staff Starts"
        case .allMeasures: return "All Measures"
        }
    }
}

private enum EndingKind: String, Hashable {
    case first = "1"
    case second = "2"

    var title: String {
        switch self {
        case .first:
            return "Ending 1"
        case .second:
            return "Ending 2"
        }
    }
}

private enum GuidedEndingSelectionPhase {
    case selectingEnding1
    case selectingEnding2
}

private struct EndingGroupSpan {
    let measureIDs: [String]
    let indexRange: ClosedRange<Int>
}

private struct GuidedEndingApplyPlan {
    let ending1IDs: [String]
    let ending2IDs: [String]
}

private struct SystemState: Decodable, Identifiable, Hashable {
    let system_id: String
    let page: Int
    let system_index: Int
    let current_value: String
    let anchor: SystemAnchor?
    let source: String?
    let manual_row_id: String?
    let staff_kind: String?

    var id: String { system_id }
}

private struct SystemAnchor: Decodable, Hashable {
    let x: Double
    let y_top: Double
    let y_bottom: Double
}

private struct MeasureState: Decodable, Hashable, Identifiable {
    let measure_id: String?
    let page: Int
    let system_id: String
    let system_index: Int?
    let measure_local_index: Int?
    let global_index: Int?
    let x_left: Double
    let x_right: Double
    let y_top: Double
    let y_bottom: Double
    let current_value: String?
    let source: String?
    let manual_row_id: String?
    let staff_kind: String?
    let excluded_from_counting: Bool?

    var id: String {
        if let measure_id, !measure_id.isEmpty {
            return measure_id
        }
        return "p\(page)_\(system_id)_m\(measure_local_index ?? -1)"
    }
}

private struct StaffBoxState: Decodable, Hashable {
    let staff_box_id: String?
    let system_id: String
    let page: Int
    let system_index: Int?
    let x_left: Double
    let x_right: Double
    let y_top: Double
    let y_bottom: Double
    let source: String?
    let in_bounds: Bool?
}

private struct MappingSummaryResponse: Decodable {
    let editable_state: EditableState?
}

private struct RelabelRequest: Encodable {
    let edits: [RelabelEdit]
}

private enum ManualStaffKind: String, CaseIterable, Identifiable, Codable {
    case single
    case grand

    var id: String { rawValue }

    var title: String {
        switch self {
        case .single: return "Single"
        case .grand: return "Grand"
        }
    }
}

private enum ManualFixTool: String, CaseIterable, Identifiable {
    case addRow
    case addMeasures
    case resizeRow
    case delete
    case exclude
    case removeLabel

    var id: String { rawValue }

    var title: String {
        switch self {
        case .addRow: return "Add Row"
        case .addMeasures: return "Add Measures"
        case .resizeRow: return "Resize Row"
        case .delete: return "Delete"
        case .exclude: return "Exclude"
        case .removeLabel: return "Remove Label"
        }
    }
}

private struct ManualRowRect: Codable, Hashable {
    var left: Double
    var right: Double
    var top: Double
    var bottom: Double
}

private struct ManualRowState: Codable, Hashable, Identifiable {
    let manualRowId: String
    let page: Int
    var staffKind: ManualStaffKind
    var rect: ManualRowRect
    var cutXs: [Double]

    enum CodingKeys: String, CodingKey {
        case manualRowId = "manual_row_id"
        case page
        case staffKind = "staff_kind"
        case rect
        case cutXs = "cut_xs"
    }

    var id: String { manualRowId }
}

private struct LabelEraseAreaState: Codable, Hashable, Identifiable, Equatable {
    let id: String
    let page: Int
    var rect: ManualRowRect
}

private struct AutoBoxState: Codable, Hashable, Identifiable {
    var measureID: String
    var left: Double
    var right: Double
    var excludedFromCounting: Bool

    enum CodingKeys: String, CodingKey {
        case measureID = "measure_id"
        case left
        case right
        case excludedFromCounting = "excluded_from_counting"
    }

    var id: String { measureID }
}

private struct AutoRowState: Codable, Hashable, Identifiable {
    let systemID: String
    let page: Int
    var rect: ManualRowRect
    var boxes: [AutoBoxState]

    enum CodingKeys: String, CodingKey {
        case systemID = "system_id"
        case page
        case rect
        case boxes
    }

    var id: String { systemID }
}

private struct ManualSelectionState: Equatable {
    let rowID: String
    let cutIndex: Int?
}

private struct AutoSelectionState: Equatable {
    let rowID: String
    let splitIndex: Int?
    let measureID: String?
}

private struct ManualEditorState: Equatable {
    let activePage: Int
    let tool: ManualFixTool
    let defaultStaffKind: ManualStaffKind
    let rows: [ManualRowState]
    let selection: ManualSelectionState?
    let pendingLabelEraseArea: LabelEraseAreaState?
}

private struct AutoEditorState: Equatable {
    let activePage: Int
    let tool: ManualFixTool
    let rows: [AutoRowState]
    let selection: AutoSelectionState?
}

private enum PendingManualFixDelete: Identifiable, Equatable {
    case manualRow(rowID: String)
    case manualLine(rowID: String, cutIndex: Int)
    case autoLine(rowID: String, splitIndex: Int)
    case autoBox(rowID: String, measureID: String)

    var id: String {
        switch self {
        case .manualRow(let rowID):
            return "manual-row-\(rowID)"
        case .manualLine(let rowID, let cutIndex):
            return "manual-line-\(rowID)-\(cutIndex)"
        case .autoLine(let rowID, let splitIndex):
            return "auto-line-\(rowID)-\(splitIndex)"
        case .autoBox(let rowID, let measureID):
            return "auto-box-\(rowID)-\(measureID)"
        }
    }
}

private struct RelabelEdit: Encodable {
    let type: String
    let system_id: String?
    let measure_id: String?
    let intValue: Int?
    let boolValue: Bool?
    let stringValue: String?
    let page: Int?
    let rows: [ManualRowState]?
    let autoRows: [AutoRowState]?
    let rect: ManualRowRect?

    enum CodingKeys: String, CodingKey {
        case type
        case system_id
        case measure_id
        case value
        case page
        case rows
        case rect
    }

    init(
        type: String,
        system_id: String? = nil,
        measure_id: String? = nil,
        intValue: Int? = nil,
        boolValue: Bool? = nil,
        stringValue: String? = nil,
        page: Int? = nil,
        rows: [ManualRowState]? = nil,
        autoRows: [AutoRowState]? = nil,
        rect: ManualRowRect? = nil
    ) {
        self.type = type
        self.system_id = system_id
        self.measure_id = measure_id
        self.intValue = intValue
        self.boolValue = boolValue
        self.stringValue = stringValue
        self.page = page
        self.rows = rows
        self.autoRows = autoRows
        self.rect = rect
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(type, forKey: .type)
        if let system_id {
            try container.encode(system_id, forKey: .system_id)
        }
        if let measure_id {
            try container.encode(measure_id, forKey: .measure_id)
        }
        if let intValue {
            try container.encode(intValue, forKey: .value)
        } else if let boolValue {
            try container.encode(boolValue, forKey: .value)
        } else if let stringValue {
            try container.encode(stringValue, forKey: .value)
        }
        if let page {
            try container.encode(page, forKey: .page)
        }
        if let rows {
            try container.encode(rows, forKey: .rows)
        } else if let autoRows {
            try container.encode(autoRows, forKey: .rows)
        }
        if let rect {
            try container.encode(rect, forKey: .rect)
        }
    }
}

private struct RelabelResult: Decodable {
    let applied_edits: [RelabelAppliedEdit]?
    let rejected_edits: [RelabelRejectedEdit]?
}

private struct RelabelAppliedEdit: Decodable {
    let type: String?
}

private struct RelabelRejectedEdit: Decodable {
    let reason: String?
}

private struct RelabelResponse: Decodable {
    let artifacts_http: ArtifactHTTP?
    let relabel: RelabelResult?
}

private struct RenderSnapshot {
    let token: UUID
    let documentLoadID: UUID
    let jobID: String
    let runID: Int?
    let pdfData: Data
    let pdfFingerprint: UInt64
    let preserveViewport: Bool
    let labelsMode: LabelsMode
    let systems: [SystemState]
    let measures: [MeasureState]
    let measureNumberOverrideIDs: Set<String>
    let restAnchorIDs: Set<String>
    let pickupAnchorIDs: Set<String>
    let aiSuggestionMeasureIDs: Set<String>
    let ending1AnchorIDs: Set<String>
    let ending2AnchorIDs: Set<String>
}

private enum EditTool: String {
    case none
    case setMeasureNumber
    case rest
    case pickup
    case ending
    case manualFix
}

private enum ActiveEditSheet: Identifiable {
    case measureNumber(MeasureState)
    case rest(MeasureState)
    case measureEdits(MeasureState)

    var id: String {
        switch self {
        case .measureNumber(let measure):
            return "measure-number-\(measure.id)"
        case .rest(let measure):
            return "rest-\(measure.id)"
        case .measureEdits(let measure):
            return "measure-edits-\(measure.id)"
        }
    }
}

// MARK: - Main View

struct ContentView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    @AppStorage("sheetmusiclabeler.persisted_job_id") private var persistedJobID: String = ""
    @AppStorage("sheetmusiclabeler.persisted_pdf_name") private var persistedPDFName: String = ""
    @AppStorage("sheetmusiclabeler.last_job_saved_at") private var persistedJobSavedAt: Double = 0
    @AppStorage("sheetmusiclabeler.accent_theme") private var accentThemeRaw: String = AccentTheme.orange.rawValue
    @AppStorage("sheetmusiclabeler.force_dark") private var forceDark: Bool = false

    private var accentTheme: AccentTheme { AccentTheme(rawValue: accentThemeRaw) ?? .orange }

    @State private var showSettings = false

    @State private var phase: AppPhase = .idle
    @State private var detailNote: String = ""
    @State private var isBusy = false

    @State private var activeJobToken: UUID = UUID()
    @State private var renderSnapshot: RenderSnapshot?
    @State private var pdfName: String = ""
    @State private var drawnOverlayCount: Int = 0

    @State private var currentJobID: String?
    @State private var currentRunID: Int?
    @State private var baselinePDFURL: String?
    @State private var correctedPDFURL: String?

    @State private var systems: [SystemState] = []
    @State private var measures: [MeasureState] = []
    @State private var labelsMode: LabelsMode = .allMeasures
    @State private var overlayGeometryWarning: String = ""

    @State private var activeEditSheet: ActiveEditSheet?
    @State private var measureEditValue: String = ""
    @State private var restEditValue: String = ""
    @State private var measureNumberOverrideValues: [String: Int] = [:]
    @State private var restMeasureCounts: [String: Int] = [:]
    @State private var pickupMeasureIDs: Set<String> = []
    @State private var aiSuggestions: AISuggestionsState?
    @State private var isReviewingAISuggestions = false
    @State private var currentAISuggestionMeasureID: String?
    @State private var isGeneratingAISuggestions = false
    @State private var aiSuggestRun: AISuggestRunState?
    @State private var endingMeasureKinds: [String: EndingKind] = [:]
    @State private var guidedEndingSelectionPhase: GuidedEndingSelectionPhase = .selectingEnding1
    @State private var pendingEnding1MeasureIDs: Set<String> = []
    @State private var pendingEnding2MeasureIDs: Set<String> = []
    @State private var activeEditTool: EditTool = .none
    @State private var manualRows: [ManualRowState] = []
    @State private var manualDraftRows: [ManualRowState] = []
    @State private var manualDraftPage: Int?
    @State private var manualFixTool: ManualFixTool = .addRow
    @State private var manualStaffKind: ManualStaffKind = .single
    @State private var manualSelection: ManualSelectionState?
    @State private var autoDraftRows: [AutoRowState] = []
    @State private var autoSelection: AutoSelectionState?
    @State private var pendingManualFixDelete: PendingManualFixDelete?
    @State private var pendingLabelEraseArea: LabelEraseAreaState?
    @State private var currentVisiblePDFPage: Int = 1
    @State private var pendingAutoScrollToTools = false
    @State private var pendingAutoScrollToPDF = false

    @State private var actionError: String?

    @State private var isServerReachable: Bool? = nil
    private let editToolsAnchorID = "edit-tools-anchor"
    private let pdfCardAnchorID = "pdf-card-anchor"
    private var isCompactWidth: Bool { horizontalSizeClass == .compact }
    private var contentHorizontalPadding: CGFloat { isCompactWidth ? 12 : 16 }
    private var cardPadding: CGFloat { isCompactWidth ? 10 : 12 }

    private var colorKeyColumns: [GridItem] {
        if isCompactWidth {
            return [
                GridItem(.flexible(minimum: 0, maximum: .infinity), spacing: 10, alignment: .leading),
                GridItem(.flexible(minimum: 0, maximum: .infinity), spacing: 10, alignment: .leading)
            ]
        }
        return [
            GridItem(.flexible(minimum: 0, maximum: .infinity), spacing: 12, alignment: .leading),
            GridItem(.flexible(minimum: 0, maximum: .infinity), spacing: 12, alignment: .leading),
            GridItem(.flexible(minimum: 0, maximum: .infinity), spacing: 12, alignment: .leading)
        ]
    }

    private var editToolColumns: [GridItem] {
        if isCompactWidth {
            return [
                GridItem(.flexible(minimum: 0, maximum: .infinity), spacing: 10, alignment: .leading),
                GridItem(.flexible(minimum: 0, maximum: .infinity), spacing: 10, alignment: .leading)
            ]
        }
        return [
            GridItem(.adaptive(minimum: 140), spacing: 10, alignment: .leading)
        ]
    }

    private var connectionStatusText: String {
        if isServerReachable == true { return "Connected" }
        if isServerReachable == false { return "Not Connected" }
        return "Checking…"
    }

    private var aiSuggestStatus: String {
        aiSuggestRun?.status?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "idle"
    }

    private var isAISuggestRunning: Bool {
        aiSuggestStatus == "running"
    }

    private var isAISuggestCompleted: Bool {
        aiSuggestStatus == "completed"
    }

    private var isAISuggestFailed: Bool {
        aiSuggestStatus == "failed"
    }

    private var aiSuggestProgressText: String? {
        guard isAISuggestRunning else { return nil }
        let total = max(0, aiSuggestRun?.systems_total ?? 0)
        let completed = max(0, aiSuggestRun?.systems_completed ?? 0)
        guard total > 0 else { return nil }
        return "System \(min(completed + 1, total)) of \(total)"
    }

    private var unresolvedAISuggestionEntries: [String: AISuggestionEntry] {
        guard isAISuggestCompleted else { return [:] }
        return aiSuggestions?.by_measure_id ?? [:]
    }

    private var unresolvedAISuggestionIDs: Set<String> {
        Set(unresolvedAISuggestionEntries.keys)
    }

    private var orderedAISuggestionMeasureIDs: [String] {
        let unresolved = unresolvedAISuggestionIDs
        guard !unresolved.isEmpty else { return [] }
        let orderedMeasures = measures.sorted { lhs, rhs in
            if lhs.page != rhs.page { return lhs.page < rhs.page }
            let lhsSystemIndex = lhs.system_index ?? 0
            let rhsSystemIndex = rhs.system_index ?? 0
            if lhsSystemIndex != rhsSystemIndex { return lhsSystemIndex < rhsSystemIndex }
            let lhsLocalIndex = lhs.measure_local_index ?? 0
            let rhsLocalIndex = rhs.measure_local_index ?? 0
            if lhsLocalIndex != rhsLocalIndex { return lhsLocalIndex < rhsLocalIndex }
            if lhs.x_left != rhs.x_left { return lhs.x_left < rhs.x_left }
            return lhs.id < rhs.id
        }
        return orderedMeasures.compactMap { measure in
            let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return unresolved.contains(measureID) ? measureID : nil
        }
    }

    private var currentAISuggestionEntry: AISuggestionEntry? {
        guard let measureID = currentAISuggestionMeasureID else { return nil }
        return unresolvedAISuggestionEntries[measureID]
    }

    private var currentAISuggestionIndex: Int? {
        guard let measureID = currentAISuggestionMeasureID else { return nil }
        return orderedAISuggestionMeasureIDs.firstIndex(of: measureID)
    }

    private var shouldLockWholeScreenForManualMeasures: Bool {
        activeEditTool == .manualFix && manualFixTool == .addMeasures
    }

    // MARK: Body

    var body: some View {
        GeometryReader { proxy in
            ScrollViewReader { scrollProxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        header
                        connectionBanner
                        controls
                        colorKeyCard
                        pdfCard(height: pdfCardHeight(for: proxy.size))
                            .id(pdfCardAnchorID)
                        editToolsCard
                            .id(editToolsAnchorID)
                    }
                    .padding(.horizontal, contentHorizontalPadding)
                    .padding(.vertical, 12)
                }
                .scrollDisabled(shouldLockWholeScreenForManualMeasures)
                .onChange(of: pendingAutoScrollToTools) { _, shouldScroll in
                    guard shouldScroll else { return }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                        withAnimation(.easeInOut(duration: 0.35)) {
                            scrollProxy.scrollTo(editToolsAnchorID, anchor: .bottom)
                        }
                        pendingAutoScrollToTools = false
                    }
                }
                .onChange(of: pendingAutoScrollToPDF) { _, shouldScroll in
                    guard shouldScroll else { return }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.10) {
                        withAnimation(.easeInOut(duration: 0.35)) {
                            scrollProxy.scrollTo(pdfCardAnchorID, anchor: .top)
                        }
                        pendingAutoScrollToPDF = false
                    }
                }
            }
        }
        .sheet(item: $activeEditSheet, onDismiss: {
            measureEditValue = ""
            restEditValue = ""
            activeEditTool = .none
        }) { sheet in
            switch sheet {
            case .measureNumber(let measure):
                EditMeasureSheet(
                    measure: measure,
                    draftValue: $measureEditValue,
                    isBusy: isBusy,
                    onApply: {
                        Task { await applySetMeasureNumber(measure: measure) }
                    }
                )
                .interactiveDismissDisabled(isBusy)
                .presentationDetents(measureNumberSheetDetents)
                .presentationDragIndicator(.visible)
                .onAppear {
                    measureEditValue = measure.current_value ?? ""
                }

            case .rest(let measure):
                EditRestSheet(
                    measure: measure,
                    currentRestCount: savedRestCount(for: measure),
                    draftValue: $restEditValue,
                    isBusy: isBusy,
                    onApply: {
                        Task { await applyRest(measure: measure) }
                    }
                )
                .interactiveDismissDisabled(isBusy)
                .presentationDetents(restSheetDetents)
                .presentationDragIndicator(.visible)
                .onAppear {
                    restEditValue = savedRestCount(for: measure).map(String.init) ?? ""
                }

            case .measureEdits(let measure):
                MeasureEditsSheet(
                    measure: measure,
                    savedEdits: savedEdits(for: measure),
                    isBusy: isBusy,
                    onChangeMeasureNumber: {
                        activeEditSheet = .measureNumber(measure)
                    },
                    onRemoveMeasureNumber: {
                        Task { await clearMeasureNumberOverride(measure: measure) }
                    },
                    onChangeRest: {
                        activeEditSheet = .rest(measure)
                    },
                    onRemoveRest: {
                        Task { await clearRest(measure: measure) }
                    },
                    onRemovePickup: {
                        Task { await clearPickup(measure: measure) }
                    }
                )
                .interactiveDismissDisabled(isBusy)
                .presentationDetents(measureEditsSheetDetents(for: measure))
                .presentationDragIndicator(.visible)
            }
        }
        .task {
            await checkServerConnectivity()
            guard !FrontendDebugConfig.disableAutoResume else { return }
            await resumePersistedJobIfPossible()
        }
        .alert("Error", isPresented: Binding(get: {
            actionError != nil
        }, set: { newValue in
            if !newValue { actionError = nil }
        })) {
            Button("OK", role: .cancel) { actionError = nil }
        } message: {
            Text(actionError ?? "")
        }
        .confirmationDialog(
            pendingManualFixDeleteTitle,
            isPresented: Binding(
                get: { pendingManualFixDelete != nil },
                set: { newValue in
                    if !newValue { pendingManualFixDelete = nil }
                }
            ),
            titleVisibility: .visible
        ) {
            Button(pendingManualFixDeleteButtonTitle, role: .destructive) {
                performPendingManualFixDelete()
            }
            Button("Cancel", role: .cancel) {
                pendingManualFixDelete = nil
            }
        } message: {
            Text(pendingManualFixDeleteMessage)
        }
        .onChange(of: labelsMode) { _, mode in
            guard !isBusy else { return }
            guard currentJobID != nil else { return }
            Task { await applyLabelsMode(mode) }
        }
        .onChange(of: manualFixTool) { _, tool in
            normalizeManualSelection(for: tool)
        }
        .sheet(isPresented: $showSettings) {
            SettingsSheet(accentThemeRaw: $accentThemeRaw, forceDark: $forceDark)
        }
        .tint(accentTheme.color)
        .preferredColorScheme(forceDark ? .dark : nil)
    }

    // MARK: View Builders

    private var connectionBanner: some View {
        Group {
            if isCompactWidth && isServerReachable == false {
                VStack(alignment: .leading, spacing: 8) {
                    bannerStatusRow
                    HStack {
                        Spacer(minLength: 0)
                        connectionBannerButtons
                    }
                }
            } else {
                HStack(spacing: 8) {
                    bannerStatusRow
                    Spacer(minLength: 0)
                    connectionBannerButtons
                }
            }
        }
        .padding(.horizontal, cardPadding)
        .padding(.vertical, isCompactWidth ? 10 : 8)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill((forceDark ? Color(.systemGray5) : Color(.secondarySystemBackground)).mix(with: accentTheme.color, by: accentTheme.backgroundTint))
        )
    }

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Sheet Music Labeler")
                    .font(.title2.weight(.semibold))
                Text("Upload a PDF, choose an edit tool, then tap a measure box.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 12)
            Button {
                showSettings = true
            } label: {
                Image(systemName: "gearshape.fill")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            .frame(minWidth: 44, minHeight: 44)
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Label Mode", selection: $labelsMode) {
                ForEach(LabelsMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .disabled(isBusy || currentJobID == nil)

            if isCompactWidth {
                VStack(alignment: .leading, spacing: 8) {
                    generateAISuggestionsButton
                    if !orderedAISuggestionMeasureIDs.isEmpty {
                        reviewAISuggestionsButton
                    }
                }
            } else {
                HStack(spacing: 8) {
                    generateAISuggestionsButton
                    if !orderedAISuggestionMeasureIDs.isEmpty {
                        reviewAISuggestionsButton
                    }
                }
            }
        }
    }

    private var generateAISuggestionsButton: some View {
        Button {
            Task { await generateAISuggestions() }
        } label: {
            Label(
                isAISuggestRunning ? "Generating..." : (isAISuggestFailed ? "Retry AI Suggestions" : "Generate AI Suggestions"),
                systemImage: "sparkles"
            )
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.small)
        .disabled(isBusy || currentJobID == nil || isAISuggestRunning)
    }

    private var reviewAISuggestionsButton: some View {
        Button {
            beginAISuggestionReview()
        } label: {
            Text("Review Suggestions (\(orderedAISuggestionMeasureIDs.count))")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(isBusy || orderedAISuggestionMeasureIDs.isEmpty)
    }

    private var colorKeyCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Color Key")
                .font(.headline)

            LazyVGrid(
                columns: colorKeyColumns,
                alignment: .leading,
                spacing: 10
            ) {
                ForEach(EditColorPalette.legendItems) { item in
                    HStack(spacing: 8) {
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(item.swiftFill)
                            .frame(width: 26, height: 16)
                            .overlay(
                                RoundedRectangle(cornerRadius: 6, style: .continuous)
                                    .stroke(item.swiftStroke, lineWidth: 1.2)
                            )

                        Text(":")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)

                        Text(item.title)
                            .font(.subheadline)
                            .foregroundStyle(.primary)
                            .lineLimit(isCompactWidth ? 2 : 1)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(cardPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill((forceDark ? Color(.systemGray5) : Color(.secondarySystemBackground)).mix(with: accentTheme.color, by: accentTheme.backgroundTint))
        )
    }

    private func pdfCard(height: CGFloat) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill((forceDark ? Color(.systemGray6) : Color(.tertiarySystemBackground)).mix(with: accentTheme.color, by: accentTheme.backgroundTint))

            if let snapshot = renderSnapshot {
                PDFOverlayContainer(
                    pdfData: snapshot.pdfData,
                    snapshotToken: snapshot.token,
                    documentLoadID: snapshot.documentLoadID,
                    preserveViewport: snapshot.preserveViewport,
                    systems: snapshot.systems,
                    measures: snapshot.measures,
                    measureNumberOverrideIDs: snapshot.measureNumberOverrideIDs,
                    restAnchorIDs: snapshot.restAnchorIDs,
                    pickupAnchorIDs: snapshot.pickupAnchorIDs,
                    aiSuggestionMeasureIDs: snapshot.aiSuggestionMeasureIDs,
                    ending1AnchorIDs: snapshot.ending1AnchorIDs,
                    ending2AnchorIDs: snapshot.ending2AnchorIDs,
                    pendingEnding1IDs: activeEditTool == .ending ? pendingEnding1MeasureIDs : [],
                    pendingEnding2IDs: activeEditTool == .ending ? pendingEnding2MeasureIDs : [],
                    labelsMode: snapshot.labelsMode,
                    manualEditor: activeManualFixManualEditor,
                    autoEditor: activeManualFixAutoEditor,
                    pendingLabelEraseArea: pendingLabelEraseArea,
                    onOverlayCount: { count in
                        DispatchQueue.main.async {
                            drawnOverlayCount = count
                        }
                    },
                    onVisiblePageChange: { page in
                        DispatchQueue.main.async {
                            currentVisiblePDFPage = max(1, page)
                        }
                    },
                    onManualRowsChange: { rows in
                        DispatchQueue.main.async {
                            replaceManualDraftRows(rows)
                        }
                    },
                    onManualSelectionChange: { selection in
                        DispatchQueue.main.async {
                            manualSelection = selection
                        }
                    },
                    onAutoRowsChange: { rows in
                        DispatchQueue.main.async {
                            replaceAutoDraftRows(rows)
                        }
                    },
                    onAutoSelectionChange: { selection in
                        DispatchQueue.main.async {
                            autoSelection = selection
                        }
                    },
                    onLabelEraseAreaChange: { area in
                        DispatchQueue.main.async {
                            pendingLabelEraseArea = area
                            manualSelection = nil
                            autoSelection = nil
                        }
                    },
                    onSelectMeasure: { measure in
                        if activeEditTool == .manualFix {
                            return
                        }
                        if isReviewingAISuggestions {
                            let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                            guard unresolvedAISuggestionIDs.contains(measureID) else { return }
                            currentAISuggestionMeasureID = measureID
                            return
                        }
                        if measure.excluded_from_counting == true {
                            return
                        }
                        switch activeEditTool {
                        case .setMeasureNumber:
                            activeEditSheet = .measureNumber(measure)
                        case .rest:
                            activeEditSheet = .rest(measure)
                        case .pickup:
                            Task { await applyPickup(measure: measure) }
                        case .ending:
                            toggleGuidedEndingSelection(measure: measure)
                        case .manualFix:
                            return
                        case .none:
                            guard hasSavedEdits(for: measure) else { return }
                            activeEditSheet = .measureEdits(measure)
                        }
                    }
                )
                if isAISuggestRunning {
                    ZStack {
                        Rectangle()
                            .fill(.ultraThinMaterial)
                            .opacity(0.96)
                        VStack(spacing: 10) {
                            ProgressView()
                                .controlSize(.large)
                            Text("Generating AI Suggestions...")
                                .font(.headline)
                            if let progress = aiSuggestProgressText {
                                Text(progress)
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .padding(24)
                    }
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "doc.richtext")
                        .font(.system(size: 28))
                        .foregroundStyle(.secondary)
                    Text("No PDF yet")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: height)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private var editToolsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Edit Tools")
                    .font(.headline)
                Spacer()
                if isReviewingAISuggestions {
                    Text("Reviewing")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(accentTheme.color)
                } else if activeEditTool != .none {
                    Text("Armed")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(accentTheme.color)
                }
            }

            Text(
                isReviewingAISuggestions
                ? "Review AI suggestions one by one."
                : (activeEditTool == .manualFix ? "Draw a missing row, add measures, then save." : "Tap a tool, then tap a measure.")
            )
                .font(.subheadline)
                .foregroundStyle(.secondary)

            LazyVGrid(
                columns: editToolColumns,
                alignment: .leading,
                spacing: 10
            ) {
                editToolButton(title: "Set Measure #", tool: .setMeasureNumber)
                editToolButton(title: "Rest", tool: .rest)
                editToolButton(title: "Pickup", tool: .pickup)
                editToolButton(title: "Ending", tool: .ending)
                editToolButton(title: "Manual Fix", tool: .manualFix)
            }

            if isReviewingAISuggestions {
                aiSuggestionReviewControls
            } else if activeEditTool == .manualFix {
                manualFixControls
            } else if activeEditTool == .ending {
                guidedEndingControls
            }
        }
        .padding(cardPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill((forceDark ? Color(.systemGray5) : Color(.secondarySystemBackground)).mix(with: accentTheme.color, by: accentTheme.backgroundTint))
        )
    }

    @ViewBuilder
    private func editToolButton(title: String, tool: EditTool) -> some View {
        if activeEditTool == tool {
            Button {
                deactivateEditTool(tool)
            } label: {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isBusy || currentJobID == nil)
        } else {
            Button {
                activateEditTool(tool)
            } label: {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .lineLimit(1)
                    .minimumScaleFactor(0.85)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isBusy || currentJobID == nil)
        }
    }

    private var guidedEndingControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(guidedEndingInstructionText)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Text("Ending 1: \(pendingEnding1MeasureIDs.count) selected • Ending 2: \(pendingEnding2MeasureIDs.count) selected")
                .font(.caption)
                .foregroundStyle(.secondary)

            if isCompactWidth {
                VStack(alignment: .leading, spacing: 8) {
                    guidedEndingActionButtons
                }
            } else {
                HStack(spacing: 8) {
                    guidedEndingActionButtons
                }
            }
        }
        .padding(.top, 2)
    }

    private var aiSuggestionReviewControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(aiSuggestionReviewHeaderText)
                .font(.subheadline.weight(.semibold))

            Text(aiSuggestionReviewDetailText)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if isCompactWidth {
                VStack(alignment: .leading, spacing: 8) {
                    aiSuggestionReviewButtons
                }
            } else {
                HStack(spacing: 8) {
                    aiSuggestionReviewButtons
                }
            }
        }
        .padding(.top, 2)
    }

    private var aiSuggestionReviewHeaderText: String {
        guard let index = currentAISuggestionIndex else {
            return "No AI suggestion selected"
        }
        return "Suggestion \(index + 1) of \(orderedAISuggestionMeasureIDs.count)"
    }

    private var aiSuggestionReviewDetailText: String {
        guard let entry = currentAISuggestionEntry else {
            return "Tap Review Suggestions to start."
        }
        switch entry.label {
        case "pickup":
            return "AI thinks this measure is a Pickup."
        case "multi_measure_rest":
            return "AI thinks this measure is a Multi-measure rest (\(entry.rest_count ?? 0))."
        case "uncertain":
            if let maybe = entry.maybe_label {
                if maybe == "multi_measure_rest", let count = entry.maybe_rest_count {
                    return "AI is unsure; maybe Multi-measure rest (\(count))."
                }
                if maybe == "pickup" {
                    return "AI is unsure; maybe Pickup."
                }
            }
            return "AI is unsure about this measure."
        default:
            return "AI suggestion ready for review."
        }
    }

    @ViewBuilder
    private var aiSuggestionReviewButtons: some View {
        Button("Accept") {
            Task { await acceptCurrentAISuggestion() }
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.small)
        .disabled(isBusy || !canAcceptCurrentAISuggestion)

        Button("Ignore") {
            Task { await ignoreCurrentAISuggestion() }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(isBusy || currentAISuggestionMeasureID == nil)

        Button("Done", role: .cancel) {
            endAISuggestionReview()
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(isBusy)
    }

    private var manualFixControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Editing page \(manualDraftPage ?? currentVisiblePDFPage)")
                .font(.subheadline.weight(.semibold))

            if manualFixTool == .addRow {
                Picker("Row Kind", selection: $manualStaffKind) {
                    ForEach(ManualStaffKind.allCases) { kind in
                        Text(kind.title).tag(kind)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(isBusy)
            }

            Picker("Manual Tool", selection: $manualFixTool) {
                ForEach(ManualFixTool.allCases) { tool in
                    Text(tool.title).tag(tool)
                }
            }
            .pickerStyle(.segmented)
            .disabled(isBusy)

            Text(manualFixInstructionText)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if isCompactWidth {
                VStack(alignment: .leading, spacing: 8) {
                    manualFixActionButtons
                }
            } else {
                HStack(spacing: 8) {
                    manualFixActionButtons
                }
            }
        }
        .padding(.top, 2)
    }

    @ViewBuilder
    private var manualFixActionButtons: some View {
        if manualFixTool == .addRow {
            Button("Done Row") {
                finishCurrentManualRow()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isBusy || !canFinishCurrentManualRow)
        }

        if manualFixTool == .delete {
            Button("Delete") {
                requestManualFixDelete()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isBusy || !canDeleteCurrentSelection)
        }

        if manualFixTool == .exclude {
            Button(selectedAutoBoxExcluded ? "Include" : "Exclude") {
                toggleSelectedAutoBoxExcluded()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isBusy || !canToggleSelectedAutoBoxExcluded)
        }

        if manualFixTool == .removeLabel {
            Button("Remove Label") {
                Task { await removeSelectedLabelArea() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isBusy || pendingLabelEraseArea == nil || currentJobID == nil)
        }

        Button("Save") {
            Task { await saveManualFix() }
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.small)
        .disabled(isBusy || currentJobID == nil || manualDraftPage == nil)

        Button("Cancel", role: .cancel) {
            cancelManualFix()
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(isBusy)
    }

    private var canFinishCurrentManualRow: Bool {
        guard manualFixTool == .addRow,
              let selection = manualSelection else { return false }
        return selection.cutIndex == nil
    }

    private var canDeleteManualSelection: Bool {
        guard manualFixTool == .delete else { return false }
        return manualSelection != nil
    }

    private var selectedAutoRow: AutoRowState? {
        guard let selection = autoSelection else { return nil }
        return autoDraftRows.first(where: { $0.systemID == selection.rowID })
    }

    private var selectedAutoBox: AutoBoxState? {
        guard let selection = autoSelection,
              let measureID = selection.measureID,
              let row = selectedAutoRow else { return nil }
        return row.boxes.first(where: { $0.measureID == measureID })
    }

    private var selectedAutoBoxExcluded: Bool {
        selectedAutoBox?.excludedFromCounting == true
    }

    private var canDeleteCurrentSelection: Bool {
        guard manualFixTool == .delete else { return false }
        if manualSelection != nil {
            return true
        }
        guard let selection = autoSelection else { return false }
        if selection.splitIndex != nil {
            return true
        }
        if selection.measureID != nil, let row = selectedAutoRow {
            return row.boxes.count > 1
        }
        return false
    }

    private var canToggleSelectedAutoBoxExcluded: Bool {
        guard manualFixTool == .exclude else { return false }
        return selectedAutoBox != nil
    }

    private var manualFixInstructionText: String {
        if let selection = manualSelection {
            if selection.cutIndex != nil {
                if manualFixTool == .delete {
                    return "Delete removes this line."
                }
                switch manualFixTool {
                case .addRow:
                    return "Tap a row to select it, then drag a corner handle to resize it."
                case .addMeasures:
                    return "Drag anywhere inside a row to place a new measure line."
                case .resizeRow:
                    return "Tap a row, then drag a corner handle to resize it."
                case .delete:
                    return "Delete removes this line."
                case .exclude:
                    return "Exclude only works on auto boxes."
                case .removeLabel:
                    return "Tap the unwanted label, confirm the blue box, then save."
                }
            }
            switch manualFixTool {
            case .addRow:
                return "Drag a corner handle to resize this row, then tap Done Row."
            case .addMeasures:
                return "Drag inside this row to place a new measure line."
            case .resizeRow:
                return "Tap a row, then drag a corner handle to resize it."
            case .delete:
                return "Delete removes this row. Tap a line instead if you only want to delete that line."
            case .exclude:
                return "Exclude only works on auto boxes."
            case .removeLabel:
                return "Tap the unwanted label, confirm the blue box, then save."
            }
        }
        if let selection = autoSelection {
            if selection.splitIndex != nil {
                switch manualFixTool {
                case .addMeasures:
                    return "Drag inside this auto box to add a split line."
                case .resizeRow:
                    return "Tap the auto row, then drag a corner handle to resize it."
                case .delete:
                    return "Delete removes this auto split line after confirmation."
                case .exclude:
                    return "Tap an auto box to exclude or include it."
                case .addRow:
                    return "Add Row only creates new manual rows on empty space."
                case .removeLabel:
                    return "Tap the unwanted label, confirm the blue box, then save."
                }
            }
            if selection.measureID != nil {
                switch manualFixTool {
                case .delete:
                    return "Delete removes this auto box after confirmation."
                case .exclude:
                    return "Tap Exclude or Include for this auto box."
                case .addMeasures:
                    return "Drag inside this auto box to add a split line."
                case .resizeRow:
                    return "Tap the auto row, then drag a corner handle to resize it."
                case .addRow:
                    return "Add Row only creates new manual rows on empty space."
                case .removeLabel:
                    return "Tap the unwanted label, confirm the blue box, then save."
                }
            }
            switch manualFixTool {
            case .addMeasures:
                return "Drag inside this auto row to place a new measure line."
            case .resizeRow:
                return "Tap the auto row, then drag a corner handle to resize it."
            case .delete:
                return "Tap an auto split or box, then press Delete."
            case .exclude:
                return "Tap an auto box, then Exclude or Include it."
            case .addRow:
                return "Add Row only creates new manual rows on empty space."
            case .removeLabel:
                return "Tap the unwanted label, confirm the blue box, then save."
            }
        }
        if pendingLabelEraseArea != nil {
            return "Confirm the blue box, or tap another label to move it."
        }
        switch manualFixTool {
        case .addRow:
            return "Touch and hold one corner, then drag to the opposite corner to make the box. Resize it if needed, then tap Done Row."
        case .addMeasures:
            return "Tap a row, then drag inside it to place a vertical measure line."
        case .resizeRow:
            return "Tap a row, then drag a corner handle to resize it."
        case .delete:
            return "Tap a row to delete the row, or tap a line to delete just that line."
        case .exclude:
            return "Tap an auto box to exclude it from counting or include it again."
        case .removeLabel:
            return "Tap the unwanted label, confirm the blue box, then save."
        }
    }

    private var guidedEndingInstructionText: String {
        switch guidedEndingSelectionPhase {
        case .selectingEnding1:
            return "Select all first-ending measures."
        case .selectingEnding2:
            return "Select all second-ending measures."
        }
    }

    private var activeManualFixManualEditor: ManualEditorState? {
        guard activeEditTool == .manualFix else { return nil }
        switch manualFixTool {
        case .addRow, .addMeasures, .resizeRow, .delete, .removeLabel:
            return ManualEditorState(
                activePage: manualDraftPage ?? currentVisiblePDFPage,
                tool: manualFixTool,
                defaultStaffKind: manualStaffKind,
                rows: manualDraftRows,
                selection: manualSelection,
                pendingLabelEraseArea: pendingLabelEraseArea
            )
        case .exclude:
            return nil
        }
    }

    private var activeManualFixAutoEditor: AutoEditorState? {
        guard activeEditTool == .manualFix else { return nil }
        switch manualFixTool {
        case .addMeasures, .resizeRow, .delete, .exclude:
            return AutoEditorState(
                activePage: manualDraftPage ?? currentVisiblePDFPage,
                tool: manualFixTool,
                rows: autoDraftRows,
                selection: autoSelection
            )
        case .addRow, .removeLabel:
            return nil
        }
    }

    private var pendingManualFixDeleteTitle: String {
        switch pendingManualFixDelete {
        case .manualRow:
            return "Delete row?"
        case .manualLine, .autoLine:
            return "Delete line?"
        case .autoBox:
            return "Delete box?"
        case nil:
            return "Delete?"
        }
    }

    private var pendingManualFixDeleteButtonTitle: String {
        switch pendingManualFixDelete {
        case .manualRow:
            return "Delete Row"
        case .manualLine, .autoLine:
            return "Delete Line"
        case .autoBox:
            return "Delete Box"
        case nil:
            return "Delete"
        }
    }

    private var pendingManualFixDeleteMessage: String {
        switch pendingManualFixDelete {
        case .manualRow:
            return "This will remove the whole manual row."
        case .manualLine:
            return "This will remove the selected manual split line."
        case .autoLine:
            return "This will remove the selected auto split line and merge the two boxes."
        case .autoBox:
            return "This will remove the selected auto box and merge its space into a neighbor."
        case nil:
            return ""
        }
    }

    private func activateEditTool(_ tool: EditTool) {
        if isReviewingAISuggestions {
            endAISuggestionReview()
        }
        if activeEditTool == .manualFix && tool != .manualFix {
            cancelManualFix()
        }
        if activeEditTool == .ending && tool != .ending {
            resetGuidedEndingDraft()
        }
        if tool == .ending {
            resetGuidedEndingDraft()
            pendingAutoScrollToPDF = true
        }
        if tool == .manualFix {
            beginManualFix()
            return
        }
        activeEditTool = tool
    }

    private func deactivateEditTool(_ tool: EditTool) {
        if tool == .manualFix {
            cancelManualFix()
            return
        }
        if tool == .ending {
            cancelGuidedEndingFlow()
            return
        }
        activeEditTool = .none
    }

    private func resetGuidedEndingDraft() {
        guidedEndingSelectionPhase = .selectingEnding1
        pendingEnding1MeasureIDs = []
        pendingEnding2MeasureIDs = []
    }

    private func cancelGuidedEndingFlow() {
        resetGuidedEndingDraft()
        activeEditTool = .none
    }

    private func beginManualFix() {
        guard currentJobID != nil, renderSnapshot != nil else {
            actionError = "Load a PDF first"
            return
        }
        let page = max(1, currentVisiblePDFPage)
        manualDraftPage = page
        manualDraftRows = savedManualRowsForPage(page)
        autoDraftRows = savedAutoRowsForPage(page)
        manualSelection = nil
        autoSelection = nil
        pendingLabelEraseArea = nil
        manualFixTool = .addRow
        activeEditTool = .manualFix
        pendingAutoScrollToPDF = true
    }

    private func cancelManualFix() {
        manualDraftRows = []
        autoDraftRows = []
        manualDraftPage = nil
        manualSelection = nil
        autoSelection = nil
        pendingManualFixDelete = nil
        pendingLabelEraseArea = nil
        manualFixTool = .addRow
        activeEditTool = .none
    }

    private func savedManualRowsForPage(_ page: Int) -> [ManualRowState] {
        manualRows.filter { $0.page == page }.sorted(by: manualRowSort)
    }

    private func manualRowSort(_ lhs: ManualRowState, _ rhs: ManualRowState) -> Bool {
        if lhs.page != rhs.page { return lhs.page < rhs.page }
        if lhs.rect.top != rhs.rect.top { return lhs.rect.top < rhs.rect.top }
        if lhs.rect.left != rhs.rect.left { return lhs.rect.left < rhs.rect.left }
        return lhs.manualRowId < rhs.manualRowId
    }

    private func savedAutoRowsForPage(_ page: Int) -> [AutoRowState] {
        let autoMeasures = measures.filter { $0.page == page && $0.source?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() != "manual" }
        let grouped = Dictionary(grouping: autoMeasures, by: { $0.system_id })
        return grouped.compactMap { systemID, rows in
            guard !systemID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            let ordered = rows.sorted { lhs, rhs in
                if lhs.x_left != rhs.x_left { return lhs.x_left < rhs.x_left }
                return lhs.id < rhs.id
            }
            guard let first = ordered.first else { return nil }
            let left = ordered.map(\.x_left).min() ?? first.x_left
            let right = ordered.map(\.x_right).max() ?? first.x_right
            let top = ordered.map(\.y_top).min() ?? first.y_top
            let bottom = ordered.map(\.y_bottom).max() ?? first.y_bottom
            let boxes = ordered.map { measure in
                AutoBoxState(
                    measureID: measure.id,
                    left: measure.x_left,
                    right: measure.x_right,
                    excludedFromCounting: measure.excluded_from_counting == true
                )
            }
            return AutoRowState(
                systemID: systemID,
                page: page,
                rect: ManualRowRect(left: left, right: right, top: top, bottom: bottom),
                boxes: boxes
            )
        }.sorted(by: autoRowSort)
    }

    private func autoRowSort(_ lhs: AutoRowState, _ rhs: AutoRowState) -> Bool {
        if lhs.page != rhs.page { return lhs.page < rhs.page }
        if lhs.rect.top != rhs.rect.top { return lhs.rect.top < rhs.rect.top }
        if lhs.rect.left != rhs.rect.left { return lhs.rect.left < rhs.rect.left }
        return lhs.systemID < rhs.systemID
    }

    private func replaceManualDraftRows(_ rows: [ManualRowState]) {
        manualDraftRows = rows.sorted(by: manualRowSort)
        if let selection = manualSelection {
            if !manualDraftRows.contains(where: { $0.manualRowId == selection.rowID }) {
                manualSelection = nil
            } else if let cutIndex = selection.cutIndex,
                      let row = manualDraftRows.first(where: { $0.manualRowId == selection.rowID }),
                      cutIndex >= row.cutXs.count {
                manualSelection = ManualSelectionState(rowID: selection.rowID, cutIndex: nil)
            }
        }
    }

    private func replaceAutoDraftRows(_ rows: [AutoRowState]) {
        autoDraftRows = rows.sorted(by: autoRowSort)
        if let selection = autoSelection {
            guard let row = autoDraftRows.first(where: { $0.systemID == selection.rowID }) else {
                autoSelection = nil
                return
            }
            if let splitIndex = selection.splitIndex,
               !(splitIndex >= 0 && splitIndex < max(0, row.boxes.count - 1)) {
                autoSelection = AutoSelectionState(rowID: selection.rowID, splitIndex: nil, measureID: nil)
                return
            }
            if let measureID = selection.measureID,
               !row.boxes.contains(where: { $0.measureID == measureID }) {
                autoSelection = AutoSelectionState(rowID: selection.rowID, splitIndex: nil, measureID: nil)
            }
        }
    }

    private func deleteSelectedManualItem() {
        guard let selection = manualSelection else { return }
        guard manualFixTool == .delete else {
            return
        }
        if let cutIndex = selection.cutIndex {
            replaceManualDraftRows(
                manualDraftRows.map { row in
                    guard row.manualRowId == selection.rowID else { return row }
                    var updated = row
                    guard cutIndex >= 0, cutIndex < updated.cutXs.count else { return row }
                    updated.cutXs.remove(at: cutIndex)
                    return updated
                }
            )
            manualSelection = ManualSelectionState(rowID: selection.rowID, cutIndex: nil)
            return
        }
        replaceManualDraftRows(manualDraftRows.filter { $0.manualRowId != selection.rowID })
        manualSelection = nil
    }

    private func deleteSelectedAutoLine() {
        guard let selection = autoSelection,
              let splitIndex = selection.splitIndex,
              let rowIndex = autoDraftRows.firstIndex(where: { $0.systemID == selection.rowID }) else { return }
        var row = autoDraftRows[rowIndex]
        guard splitIndex >= 0, splitIndex < row.boxes.count - 1 else { return }
        var leftBox = row.boxes[splitIndex]
        let rightBox = row.boxes[splitIndex + 1]
        leftBox.right = rightBox.right
        leftBox.excludedFromCounting = leftBox.excludedFromCounting || rightBox.excludedFromCounting
        row.boxes.removeSubrange(splitIndex...(splitIndex + 1))
        row.boxes.insert(leftBox, at: splitIndex)
        autoDraftRows[rowIndex] = row
        replaceAutoDraftRows(autoDraftRows)
        autoSelection = AutoSelectionState(rowID: row.systemID, splitIndex: nil, measureID: nil)
    }

    private func toggleSelectedAutoBoxExcluded() {
        guard let selection = autoSelection,
              let measureID = selection.measureID,
              let rowIndex = autoDraftRows.firstIndex(where: { $0.systemID == selection.rowID }),
              let boxIndex = autoDraftRows[rowIndex].boxes.firstIndex(where: { $0.measureID == measureID }) else { return }
        autoDraftRows[rowIndex].boxes[boxIndex].excludedFromCounting.toggle()
        replaceAutoDraftRows(autoDraftRows)
    }

    private func deleteSelectedAutoBox() {
        guard let selection = autoSelection,
              let measureID = selection.measureID,
              let rowIndex = autoDraftRows.firstIndex(where: { $0.systemID == selection.rowID }) else { return }
        var row = autoDraftRows[rowIndex]
        guard let boxIndex = row.boxes.firstIndex(where: { $0.measureID == measureID }),
              row.boxes.count > 1 else { return }
        if boxIndex > 0 {
            row.boxes[boxIndex - 1].right = row.boxes[boxIndex].right
            row.boxes[boxIndex - 1].excludedFromCounting = row.boxes[boxIndex - 1].excludedFromCounting || row.boxes[boxIndex].excludedFromCounting
        } else if row.boxes.count > 1 {
            row.boxes[1].left = row.boxes[0].left
            row.boxes[1].excludedFromCounting = row.boxes[1].excludedFromCounting || row.boxes[0].excludedFromCounting
        }
        row.boxes.remove(at: boxIndex)
        autoDraftRows[rowIndex] = row
        replaceAutoDraftRows(autoDraftRows)
        autoSelection = AutoSelectionState(rowID: row.systemID, splitIndex: nil, measureID: nil)
    }

    private func requestManualFixDelete() {
        guard manualFixTool == .delete else { return }
        if let selection = manualSelection {
            if let cutIndex = selection.cutIndex {
                pendingManualFixDelete = .manualLine(rowID: selection.rowID, cutIndex: cutIndex)
            } else {
                pendingManualFixDelete = .manualRow(rowID: selection.rowID)
            }
            return
        }
        guard let selection = autoSelection else { return }
        if let splitIndex = selection.splitIndex {
            pendingManualFixDelete = .autoLine(rowID: selection.rowID, splitIndex: splitIndex)
            return
        }
        if let measureID = selection.measureID,
           canDeleteCurrentSelection {
            pendingManualFixDelete = .autoBox(rowID: selection.rowID, measureID: measureID)
        }
    }

    private func performPendingManualFixDelete() {
        guard let pending = pendingManualFixDelete else { return }
        pendingManualFixDelete = nil
        switch pending {
        case .manualRow(let rowID):
            manualSelection = ManualSelectionState(rowID: rowID, cutIndex: nil)
            deleteSelectedManualItem()
        case .manualLine(let rowID, let cutIndex):
            manualSelection = ManualSelectionState(rowID: rowID, cutIndex: cutIndex)
            deleteSelectedManualItem()
        case .autoLine(let rowID, let splitIndex):
            autoSelection = AutoSelectionState(rowID: rowID, splitIndex: splitIndex, measureID: nil)
            deleteSelectedAutoLine()
        case .autoBox(let rowID, let measureID):
            autoSelection = AutoSelectionState(rowID: rowID, splitIndex: nil, measureID: measureID)
            deleteSelectedAutoBox()
        }
    }

    private func finishCurrentManualRow() {
        guard canFinishCurrentManualRow else { return }
        manualSelection = nil
    }

    private func normalizeManualSelection(for tool: ManualFixTool) {
        pendingManualFixDelete = nil
        if tool != .removeLabel {
            pendingLabelEraseArea = nil
        }
        if tool == .exclude {
            manualSelection = nil
            if let selection = autoSelection, selection.splitIndex != nil {
                autoSelection = AutoSelectionState(rowID: selection.rowID, splitIndex: nil, measureID: nil)
            }
            return
        }
        if tool == .removeLabel {
            manualSelection = nil
            autoSelection = nil
            return
        }
        guard let selection = manualSelection else { return }
        switch tool {
        case .delete:
            return
        case .addRow, .addMeasures, .resizeRow:
            if selection.cutIndex != nil {
                manualSelection = ManualSelectionState(rowID: selection.rowID, cutIndex: nil)
            }
        case .exclude, .removeLabel:
            return
        }
        if let selection = autoSelection,
           selection.splitIndex != nil || selection.measureID != nil {
            autoSelection = AutoSelectionState(rowID: selection.rowID, splitIndex: nil, measureID: nil)
        }
    }

    private func saveManualFix() async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        guard let page = manualDraftPage else {
            actionError = "Missing page"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Saving manual fixes..."
            let cleanedManualRows = manualDraftRows.map { row in
                var cleaned = row
                cleaned.cutXs = row.cutXs.sorted()
                return cleaned
            }
            let cleanedAutoRows = autoDraftRows.map { row in
                var cleaned = row
                cleaned.boxes = row.boxes.sorted { lhs, rhs in
                    if lhs.left != rhs.left { return lhs.left < rhs.left }
                    return lhs.measureID < rhs.measureID
                }
                return cleaned
            }
            var edits = [
                RelabelEdit(
                    type: "replace_manual_rows_for_page",
                    page: page,
                    rows: cleanedManualRows
                ),
                RelabelEdit(
                    type: "replace_auto_rows_for_page",
                    page: page,
                    autoRows: cleanedAutoRows
                )
            ]
            if let area = pendingLabelEraseArea {
                edits.append(
                    RelabelEdit(
                        type: "remove_label_area",
                        page: area.page,
                        rect: area.rect
                    )
                )
            }
            let relabel = try await apiRelabel(jobID: jobID, edits: edits)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )

            let refreshedPage = page
            manualDraftPage = refreshedPage
            manualDraftRows = savedManualRowsForPage(refreshedPage)
            autoDraftRows = savedAutoRowsForPage(refreshedPage)
            manualSelection = nil
            autoSelection = nil
            pendingManualFixDelete = nil
            pendingLabelEraseArea = nil
            activeEditTool = .manualFix
            phase = .ready
            detailNote = "Manual fixes saved"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func removeSelectedLabelArea() async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        guard let area = pendingLabelEraseArea else {
            actionError = "Tap a label first"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Removing label..."
            let relabel = try await apiRelabel(
                jobID: jobID,
                edits: [
                    RelabelEdit(
                        type: "remove_label_area",
                        page: area.page,
                        rect: area.rect
                    )
                ]
            )
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )

            pendingLabelEraseArea = nil
            manualDraftPage = max(1, currentVisiblePDFPage)
            manualDraftRows = savedManualRowsForPage(manualDraftPage ?? currentVisiblePDFPage)
            autoDraftRows = savedAutoRowsForPage(manualDraftPage ?? currentVisiblePDFPage)
            manualSelection = nil
            autoSelection = nil
            activeEditTool = .manualFix
            phase = .ready
            detailNote = "Label removed"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    @ViewBuilder
    private var guidedEndingActionButtons: some View {
        switch guidedEndingSelectionPhase {
        case .selectingEnding1:
            Button("Finish Ending 1") {
                finishGuidedEnding1Selection()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isBusy)

            Button("Cancel", role: .cancel) {
                cancelGuidedEndingFlow()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isBusy)

        case .selectingEnding2:
            Button("Back to Ending 1") {
                guidedEndingSelectionPhase = .selectingEnding1
                pendingAutoScrollToPDF = true
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isBusy)

            Button("Finish Applying Endings") {
                Task { await applyGuidedEndings() }
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(isBusy)

            Button("Cancel", role: .cancel) {
                cancelGuidedEndingFlow()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(isBusy)
        }
    }

    private func finishGuidedEnding1Selection() {
        guard !pendingEnding1MeasureIDs.isEmpty else {
            actionError = "Select at least one Ending 1 measure"
            return
        }
        guidedEndingSelectionPhase = .selectingEnding2
        pendingAutoScrollToPDF = true
    }

    private var phaseLabel: String {
        switch phase {
        case .idle: return "Idle"
        case .uploading: return "Uploading"
        case .dispatching: return "Dispatching"
        case .processing: return "Processing"
        case .ready: return "Ready"
        case .failed: return "Failed"
        }
    }

    private var phaseColor: Color {
        switch phase {
        case .idle: return .secondary
        case .uploading, .dispatching, .processing: return .orange
        case .ready: return .green
        case .failed: return .red
        }
    }

    // MARK: Job Lifecycle

    private func pickPDF() {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.pdf], asCopy: true)
        picker.allowsMultipleSelection = false
        picker.delegate = DocumentPickerDelegate.shared
        DocumentPickerDelegate.shared.onPick = { url in
            Task { await beginRun(from: url) }
        }
        UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first?.windows.first?.rootViewController?
            .present(picker, animated: true)
    }

    private func beginRun(from fileURL: URL) async {
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }

        let token = startNewJobToken()
        clearRuntimeForNewRun()
        pdfName = fileURL.lastPathComponent

        do {
            phase = .uploading
            detailNote = "Uploading PDF..."
            let upload = try await apiUploadPDF(fileURL: fileURL)
            guard isTokenCurrent(token) else { return }

            phase = .dispatching
            detailNote = "Creating job..."
            var seedID = String(fileURL.deletingPathExtension().lastPathComponent.prefix(32))
                .replacingOccurrences(of: "[^A-Za-z0-9_-]", with: "", options: .regularExpression)
            if seedID.isEmpty {
                seedID = "score"
            }
            if FrontendDebugConfig.debugSafeJobIDs {
                let ts = Int(Date().timeIntervalSince1970)
                let short = UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(8)
                seedID = "\(seedID)-\(ts)-\(short)"
            }
            let create = try await apiCreateJob(pdfGCSURI: upload.pdf_gcs_uri, proposedJobID: seedID)
            guard isTokenCurrent(token) else { return }

            currentJobID = create.job_id
            currentRunID = create.run_id
            persistedJobID = create.job_id
            persistedPDFName = pdfName
            persistedJobSavedAt = Date().timeIntervalSince1970

            phase = .processing
            detailNote = "Processing score..."

            let finalJob = try await pollUntilFinished(jobID: create.job_id, token: token)
            guard isTokenCurrent(token, expectedJobID: create.job_id) else { return }
            currentRunID = finalJob.run_id ?? currentRunID

            let state = try await apiFetchState(jobID: create.job_id)
            guard isTokenCurrent(token, expectedJobID: create.job_id) else { return }

            let renderedURL = try await waitForRenderedPDFURL(jobID: create.job_id, initialArtifacts: finalJob.artifacts_http, token: token)
            guard isTokenCurrent(token, expectedJobID: create.job_id) else { return }
            baselinePDFURL = renderedURL
            correctedPDFURL = finalJob.artifacts_http?.audiveris_out_corrected_pdf
            let pdfData = try await downloadPDF(urlString: renderedURL)
            guard isTokenCurrent(token, expectedJobID: create.job_id) else { return }

            let effectiveMode = labelsModeFromState(state)
            commitRenderSnapshot(
                jobID: create.job_id,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: effectiveMode,
                token: token
            )
            pendingAutoScrollToTools = true

            phase = .ready
            detailNote = "Rendered PDF ready"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func reloadRenderedPDF() async {
        guard let jobID = currentJobID else { return }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Reloading rendered PDF..."
            let job = try await apiGetJob(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            let renderedURL = try await waitForRenderedPDFURL(jobID: jobID, initialArtifacts: job.artifacts_http, token: token)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            let pdfData = try await downloadPDF(urlString: renderedURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: job.run_id ?? currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token
            )
            if phase != .ready { phase = .ready }
            detailNote = "Rendered PDF ready"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func applySetMeasureNumber(measure: MeasureState) async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else {
            actionError = "Missing measure ID"
            return
        }
        guard let newValue = Int(measureEditValue.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            actionError = "Enter a valid number"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Applying edit..."
            let relabel = try await apiRelabel(jobID: jobID, edits: [
                RelabelEdit(type: "set_measure_number", system_id: nil, measure_id: measureID, intValue: newValue, boolValue: nil, stringValue: nil)
            ])
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )
            activeEditSheet = nil

            phase = .ready
            detailNote = "Edit applied"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func applyRest(measure: MeasureState) async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else {
            actionError = "Missing measure ID"
            return
        }
        guard let restCount = Int(restEditValue.trimmingCharacters(in: .whitespacesAndNewlines)), restCount >= 1 else {
            actionError = "Enter a valid rest count"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Applying rest..."
            let relabel = try await apiRelabel(jobID: jobID, edits: [
                RelabelEdit(type: "set_rest_measure", system_id: nil, measure_id: measureID, intValue: restCount, boolValue: nil, stringValue: nil)
            ])
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )
            activeEditSheet = nil

            phase = .ready
            detailNote = "Rest applied"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func clearMeasureNumberOverride(measure: MeasureState) async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else {
            actionError = "Missing measure ID"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Removing measure number..."
            let relabel = try await apiRelabel(jobID: jobID, edits: [
                RelabelEdit(type: "clear_measure_number", system_id: nil, measure_id: measureID, intValue: nil, boolValue: nil, stringValue: nil)
            ])
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )
            activeEditSheet = nil

            phase = .ready
            detailNote = "Measure number removed"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func clearRest(measure: MeasureState) async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else {
            actionError = "Missing measure ID"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Removing rest..."
            let relabel = try await apiRelabel(jobID: jobID, edits: [
                RelabelEdit(type: "set_rest_measure", system_id: nil, measure_id: measureID, intValue: 0, boolValue: nil, stringValue: nil)
            ])
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )
            activeEditSheet = nil

            phase = .ready
            detailNote = "Rest removed"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func clearPickup(measure: MeasureState) async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else {
            actionError = "Missing measure ID"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Removing pickup..."
            let relabel = try await apiRelabel(jobID: jobID, edits: [
                RelabelEdit(type: "set_pickup_measure", system_id: nil, measure_id: measureID, intValue: nil, boolValue: false, stringValue: nil)
            ])
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )
            activeEditSheet = nil

            phase = .ready
            detailNote = "Pickup removed"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func resumePersistedJobIfPossible() async {
        let saved = persistedJobID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !saved.isEmpty else { return }

        let now = Date().timeIntervalSince1970
        let age = now - persistedJobSavedAt
        if persistedJobSavedAt <= 0 || age > 24 * 60 * 60 {
            clearSession()
            return
        }

        guard !isBusy else { return }
        let token = startNewJobToken()
        isBusy = true
        defer { isBusy = false }

        do {
            phase = .processing
            detailNote = "Resuming previous job..."
            currentJobID = saved
            pdfName = persistedPDFName

            let job = try await apiGetJob(jobID: saved)
            guard isTokenCurrent(token, expectedJobID: saved) else { return }
            currentRunID = job.run_id

            if (job.status ?? "") != "succeeded" {
                let final = try await pollUntilFinished(jobID: saved, token: token)
                guard isTokenCurrent(token, expectedJobID: saved) else { return }
                currentRunID = final.run_id ?? currentRunID
            }

            let state = try await apiFetchState(jobID: saved)
            guard isTokenCurrent(token, expectedJobID: saved) else { return }

            let renderedURL = try await waitForRenderedPDFURL(jobID: saved, initialArtifacts: job.artifacts_http, token: token)
            guard isTokenCurrent(token, expectedJobID: saved) else { return }
            baselinePDFURL = renderedURL
            let pdfData = try await downloadPDF(urlString: renderedURL)
            guard isTokenCurrent(token, expectedJobID: saved) else { return }
            commitRenderSnapshot(
                jobID: saved,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token
            )

            if (state.ai_suggest_run?.status ?? "").lowercased() == "running" {
                try await continueAISuggestionRun(jobID: saved, token: token)
                return
            }

            phase = .ready
            detailNote = "Rendered PDF ready"
            persistedJobSavedAt = Date().timeIntervalSince1970
        } catch is CancellationError {
            return
        } catch {
            let msg = error.localizedDescription.lowercased()
            if msg.contains("unknown job_id") || msg.contains("not found") || msg.contains("artifacts not found") || msg.contains("per-run artifacts not found") || msg.contains("stale") {
                clearSession()
                return
            }
            phase = .failed
            detailNote = error.localizedDescription
        }
    }

    private func waitForRenderedPDFURL(jobID: String, initialArtifacts: ArtifactHTTP?, token: UUID? = nil) async throws -> String {
        if let token, !isTokenCurrent(token, expectedJobID: jobID) {
            throw CancellationError()
        }
        if let url = initialArtifacts?.audiveris_out_pdf?.nonEmpty {
            return url
        }
        for _ in 0..<BackendConfig.artifactRetryAttempts {
            try await Task.sleep(nanoseconds: BackendConfig.artifactRetryDelaySeconds * 1_000_000_000)
            if let token, !isTokenCurrent(token, expectedJobID: jobID) {
                throw CancellationError()
            }
            let job = try await apiGetJob(jobID: jobID)
            if let url = job.artifacts_http?.audiveris_out_pdf?.nonEmpty {
                return url
            }
        }
        throw LocalError("Rendered PDF not ready")
    }

    private func pollUntilFinished(jobID: String, token: UUID? = nil) async throws -> JobStatusResponse {
        var attempts = 0
        while attempts < BackendConfig.maxPollAttempts {
            attempts += 1
            if let token, !isTokenCurrent(token, expectedJobID: jobID) {
                throw CancellationError()
            }
            let job = try await apiGetJob(jobID: jobID)
            let status = (job.status ?? "").lowercased()

            if status == "succeeded" {
                return job
            }
            if status == "failed" || status == "cancelled" {
                throw LocalError("Processing failed")
            }

            try await Task.sleep(nanoseconds: BackendConfig.pollSeconds * 1_000_000_000)
        }
        throw LocalError("Timed out while processing")
    }

    private var canAcceptCurrentAISuggestion: Bool {
        guard let entry = currentAISuggestionEntry else { return false }
        switch entry.label {
        case "pickup":
            return true
        case "multi_measure_rest":
            return (entry.rest_count ?? 0) > 0
        case "uncertain":
            if entry.maybe_label == "pickup" {
                return true
            }
            if entry.maybe_label == "multi_measure_rest" {
                return (entry.maybe_rest_count ?? 0) > 0
            }
            return false
        default:
            return false
        }
    }

    private func beginAISuggestionReview(startingAt preferredMeasureID: String? = nil) {
        if activeEditTool == .ending {
            resetGuidedEndingDraft()
        }
        activeEditTool = .none
        isReviewingAISuggestions = true
        syncAISuggestionReviewSelection(preferredIndex: nil, preferredMeasureID: preferredMeasureID)
        pendingAutoScrollToPDF = true
    }

    private func endAISuggestionReview() {
        isReviewingAISuggestions = false
        currentAISuggestionMeasureID = nil
    }

    private func syncAISuggestionReviewSelection(preferredIndex: Int? = nil, preferredMeasureID: String? = nil) {
        let ordered = orderedAISuggestionMeasureIDs
        guard !ordered.isEmpty else {
            currentAISuggestionMeasureID = nil
            isReviewingAISuggestions = false
            return
        }
        if let preferredMeasureID, ordered.contains(preferredMeasureID) {
            currentAISuggestionMeasureID = preferredMeasureID
            return
        }
        if let current = currentAISuggestionMeasureID, ordered.contains(current) {
            return
        }
        if let preferredIndex {
            currentAISuggestionMeasureID = ordered[min(max(preferredIndex, 0), ordered.count - 1)]
            return
        }
        if isReviewingAISuggestions {
            currentAISuggestionMeasureID = ordered.first
        }
    }

    private func aiSuggestionRelabelEdit(measureID: String, entry: AISuggestionEntry) -> RelabelEdit? {
        switch entry.label {
        case "pickup":
            return RelabelEdit(type: "set_pickup_measure", system_id: nil, measure_id: measureID, intValue: nil, boolValue: true, stringValue: nil)
        case "multi_measure_rest":
            guard let restCount = entry.rest_count, restCount > 0 else { return nil }
            return RelabelEdit(type: "set_rest_measure", system_id: nil, measure_id: measureID, intValue: restCount, boolValue: nil, stringValue: nil)
        case "uncertain":
            if entry.maybe_label == "pickup" {
                return RelabelEdit(type: "set_pickup_measure", system_id: nil, measure_id: measureID, intValue: nil, boolValue: true, stringValue: nil)
            }
            if entry.maybe_label == "multi_measure_rest", let count = entry.maybe_rest_count, count > 0 {
                return RelabelEdit(type: "set_rest_measure", system_id: nil, measure_id: measureID, intValue: count, boolValue: nil, stringValue: nil)
            }
            return nil
        default:
            return nil
        }
    }

    private func refreshAISuggestionOverlay(
        using suggestions: AISuggestionsState?,
        run: AISuggestRunState?,
        preferredIndex: Int? = nil,
        preferredMeasureID: String? = nil
    ) {
        aiSuggestions = suggestions
        aiSuggestRun = run
        if let snapshot = renderSnapshot {
            let refreshed = RenderSnapshot(
                token: UUID(),
                documentLoadID: snapshot.documentLoadID,
                jobID: snapshot.jobID,
                runID: snapshot.runID,
                pdfData: snapshot.pdfData,
                pdfFingerprint: snapshot.pdfFingerprint,
                preserveViewport: snapshot.preserveViewport,
                labelsMode: snapshot.labelsMode,
                systems: snapshot.systems,
                measures: snapshot.measures,
                measureNumberOverrideIDs: snapshot.measureNumberOverrideIDs,
                restAnchorIDs: snapshot.restAnchorIDs,
                pickupAnchorIDs: snapshot.pickupAnchorIDs,
                aiSuggestionMeasureIDs: normalizedAISuggestionMeasureIDs(suggestions, run: run),
                ending1AnchorIDs: snapshot.ending1AnchorIDs,
                ending2AnchorIDs: snapshot.ending2AnchorIDs
            )
            renderSnapshot = refreshed
        }
        syncAISuggestionReviewSelection(preferredIndex: preferredIndex, preferredMeasureID: preferredMeasureID)
    }

    private func aiSuggestFailureMessage(_ response: AISuggestResponse?) -> String {
        response?.error?.message
            ?? response?.ai_suggest_run?.last_error?.message
            ?? aiSuggestRun?.last_error?.message
            ?? "AI suggestions stopped before finishing."
    }

    private func finalizeAISuggestionRun() {
        phase = .ready
        if orderedAISuggestionMeasureIDs.isEmpty {
            detailNote = "No AI suggestions found"
            endAISuggestionReview()
        } else {
            detailNote = "AI suggestions ready"
            beginAISuggestionReview()
        }
    }

    private func continueAISuggestionRun(jobID: String, token: UUID) async throws {
        while true {
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }

            if isAISuggestCompleted {
                finalizeAISuggestionRun()
                return
            }

            if isAISuggestFailed {
                phase = .ready
                detailNote = "AI suggestions stopped before finishing."
                actionError = aiSuggestRun?.last_error?.message ?? "AI suggestions stopped before finishing."
                endAISuggestionReview()
                return
            }

            detailNote = aiSuggestProgressText ?? "Generating AI suggestions..."

            do {
                let response = try await apiStepAISuggestions(jobID: jobID)
                guard isTokenCurrent(token, expectedJobID: jobID) else { return }
                currentRunID = response.run_id ?? currentRunID
                refreshAISuggestionOverlay(using: response.ai_suggestions, run: response.ai_suggest_run)

                if (response.ai_suggest_run?.status ?? "").lowercased() == "failed" {
                    phase = .ready
                    detailNote = "AI suggestions stopped before finishing."
                    actionError = aiSuggestFailureMessage(response)
                    endAISuggestionReview()
                    return
                }

                if (response.ai_suggest_run?.status ?? "").lowercased() == "completed" {
                    finalizeAISuggestionRun()
                    return
                }
            } catch {
                guard isTokenCurrent(token, expectedJobID: jobID) else { return }
                if let state = try? await apiFetchState(jobID: jobID) {
                    guard isTokenCurrent(token, expectedJobID: jobID) else { return }
                    refreshAISuggestionOverlay(using: state.ai_suggestions, run: state.ai_suggest_run)

                    if isAISuggestCompleted {
                        finalizeAISuggestionRun()
                        return
                    }
                    if isAISuggestFailed {
                        phase = .ready
                        detailNote = "AI suggestions stopped before finishing."
                        actionError = aiSuggestRun?.last_error?.message ?? "AI suggestions stopped before finishing."
                        endAISuggestionReview()
                        return
                    }
                    if isAISuggestRunning {
                        continue
                    }
                }
                throw error
            }
        }
    }

    private func generateAISuggestions() async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        isGeneratingAISuggestions = true
        defer {
            isGeneratingAISuggestions = false
            isBusy = false
        }

        do {
            phase = .processing
            detailNote = "Generating AI suggestions..."
            let response = try await apiGenerateAISuggestions(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            currentRunID = response.run_id ?? currentRunID
            refreshAISuggestionOverlay(using: response.ai_suggestions, run: response.ai_suggest_run)

            if (response.ai_suggest_run?.status ?? "").lowercased() == "completed" {
                finalizeAISuggestionRun()
                return
            }
            if (response.ai_suggest_run?.status ?? "").lowercased() == "failed" {
                phase = .ready
                detailNote = "AI suggestions stopped before finishing."
                actionError = aiSuggestFailureMessage(response)
                endAISuggestionReview()
                return
            }

            try await continueAISuggestionRun(jobID: jobID, token: token)
        } catch is CancellationError {
            return
        } catch {
            phase = .ready
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func ignoreCurrentAISuggestion() async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        guard let measureID = currentAISuggestionMeasureID,
              let currentIndex = currentAISuggestionIndex else {
            actionError = "No AI suggestion selected"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Ignoring AI suggestion..."
            let response = try await apiDismissAISuggestion(jobID: jobID, measureID: measureID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            currentRunID = response.run_id ?? currentRunID
            refreshAISuggestionOverlay(using: response.ai_suggestions, run: aiSuggestRun, preferredIndex: currentIndex)
            phase = .ready
            detailNote = "Suggestion ignored"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func acceptCurrentAISuggestion() async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        guard let measureID = currentAISuggestionMeasureID,
              let entry = currentAISuggestionEntry,
              let currentIndex = currentAISuggestionIndex,
              let edit = aiSuggestionRelabelEdit(measureID: measureID, entry: entry) else {
            actionError = "This AI suggestion cannot be accepted directly"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Applying AI suggestion..."
            let relabel = try await apiRelabel(jobID: jobID, edits: [edit])
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )
            syncAISuggestionReviewSelection(preferredIndex: currentIndex)
            phase = .ready
            detailNote = "AI suggestion applied"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    // MARK: API Client

    private func apiUploadPDF(fileURL: URL) async throws -> UploadResponse {
        let endpoint = BackendConfig.baseURL.appending(path: "api/omr/uploads")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        let data = try Data(contentsOf: fileURL)
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: application/pdf\r\n\r\n".data(using: .utf8)!)
        body.append(data)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body
        let (responseData, response) = try await URLSession.shared.data(for: request)
        try validateHTTP(response: response, data: responseData)
        return try JSONDecoder().decode(UploadResponse.self, from: responseData)
    }

    private func apiCreateJob(pdfGCSURI: String, proposedJobID: String) async throws -> CreateJobResponse {
        let endpoint = BackendConfig.baseURL.appending(path: "api/omr/jobs")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let payload = CreateJobRequest(pdf_gcs_uri: pdfGCSURI, job_id: proposedJobID)
        request.httpBody = try JSONEncoder().encode(payload)

        let (data, response) = try await URLSession.shared.data(for: request)
        try validateHTTP(response: response, data: data)
        return try JSONDecoder().decode(CreateJobResponse.self, from: data)
    }

    private func apiGetJob(jobID: String) async throws -> JobStatusResponse {
        let endpoint = BackendConfig.baseURL.appending(path: "api/omr/jobs/\(jobID)")
        let (data, response) = try await URLSession.shared.data(from: endpoint)
        try validateHTTP(response: response, data: data)
        return try JSONDecoder().decode(JobStatusResponse.self, from: data)
    }

    private func apiFetchState(jobID: String) async throws -> JobStateResponse {
        let endpoint = BackendConfig.baseURL.appending(path: "api/omr/jobs/\(jobID)/state")
        let (data, response) = try await URLSession.shared.data(from: endpoint)
        try validateHTTP(response: response, data: data)
        return try JSONDecoder().decode(JobStateResponse.self, from: data)
    }

    private func apiGenerateAISuggestions(jobID: String) async throws -> AISuggestResponse {
        let endpoint = BackendConfig.baseURL.appending(path: "api/omr/jobs/\(jobID)/ai-suggest")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"

        let (data, response) = try await URLSession.shared.data(for: request)
        try validateHTTP(response: response, data: data)
        return try JSONDecoder().decode(AISuggestResponse.self, from: data)
    }

    private func apiStepAISuggestions(jobID: String) async throws -> AISuggestResponse {
        let endpoint = BackendConfig.baseURL.appending(path: "api/omr/jobs/\(jobID)/ai-suggest/step")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"

        let (data, response) = try await URLSession.shared.data(for: request)
        try validateHTTP(response: response, data: data)
        return try JSONDecoder().decode(AISuggestResponse.self, from: data)
    }

    private func apiDismissAISuggestion(jobID: String, measureID: String) async throws -> AIDismissResponse {
        let endpoint = BackendConfig.baseURL.appending(path: "api/omr/jobs/\(jobID)/ai-suggestions/\(measureID)/dismiss")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"

        let (data, response) = try await URLSession.shared.data(for: request)
        try validateHTTP(response: response, data: data)
        return try JSONDecoder().decode(AIDismissResponse.self, from: data)
    }

    private func apiRelabel(jobID: String, edits: [RelabelEdit]) async throws -> RelabelResponse {
        let endpoint = BackendConfig.baseURL.appending(path: "api/omr/jobs/\(jobID)/relabel")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(RelabelRequest(edits: edits))

        let (data, response) = try await URLSession.shared.data(for: request)
        try validateHTTP(response: response, data: data)
        return try JSONDecoder().decode(RelabelResponse.self, from: data)
    }

    private func validateRelabelOutcome(_ response: RelabelResponse) throws {
        let appliedCount = response.relabel?.applied_edits?.count ?? 0
        let rejected = response.relabel?.rejected_edits ?? []
        guard appliedCount == 0, let first = rejected.first else { return }
        let raw = first.reason?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "edit_rejected"
        let message = raw
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
        throw LocalError(message)
    }

    // MARK: Helpers

    private func checkServerConnectivity() async {
        do {
            let (_, response) = try await URLSession.shared.data(from: BackendConfig.baseURL)
            let ok = (response as? HTTPURLResponse).map { (200...299).contains($0.statusCode) } ?? false
            await MainActor.run { isServerReachable = ok }
        } catch {
            await MainActor.run { isServerReachable = false }
        }
    }

    private func labelsModeFromState(_ state: JobStateResponse) -> LabelsMode {
        LabelsMode(rawValue: state.editable_state.labels_mode?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "")
            ?? .allMeasures
    }

    private func startNewJobToken() -> UUID {
        let token = UUID()
        activeJobToken = token
        return token
    }

    private func isTokenCurrent(_ token: UUID, expectedJobID: String? = nil) -> Bool {
        guard token == activeJobToken else { return false }
        if let expectedJobID,
           let currentJobID,
           currentJobID != expectedJobID {
            return false
        }
        return true
    }

    private func pdfFingerprint(_ data: Data) -> UInt64 {
        var value: UInt64 = UInt64(data.count)
        for b in data.prefix(256) {
            value = (value &* 1099511628211) ^ UInt64(b)
        }
        return value
    }

    private func commitRenderSnapshot(
        jobID: String,
        runID: Int?,
        pdfData: Data,
        editable: EditableState,
        aiSuggestions: AISuggestionsState?,
        aiSuggestRun: AISuggestRunState?,
        labelsMode: LabelsMode,
        token: UUID,
        preserveViewport: Bool = false
    ) {
        guard isTokenCurrent(token, expectedJobID: jobID) else { return }
        let normalizedMeasures = editable.measures ?? []
        let normalizedMeasureOverrides = normalizedMeasureNumberOverrides(editable.measure_number_overrides)
        let overrideIDs = Set(normalizedMeasureOverrides.keys)
        let normalizedRestCounts = normalizedRestMeasureCounts(editable.rest_measures)
        let normalizedPickupIDs = normalizedPickupMeasureIDs(editable.pickup_measures)
        let normalizedAISuggestionIDs = normalizedAISuggestionMeasureIDs(aiSuggestions, run: aiSuggestRun)
        let normalizedEndingKinds = normalizedEndingMeasureKinds(editable.endings)
        let normalizedManualRows = normalizedManualRows(editable.manual_rows)
        let ending1IDs = Set(normalizedEndingKinds.compactMap { $0.value == .first ? $0.key : nil })
        let ending2IDs = Set(normalizedEndingKinds.compactMap { $0.value == .second ? $0.key : nil })
        let snapshot = RenderSnapshot(
            token: token,
            documentLoadID: UUID(),
            jobID: jobID,
            runID: runID,
            pdfData: pdfData,
            pdfFingerprint: pdfFingerprint(pdfData),
            preserveViewport: preserveViewport,
            labelsMode: labelsMode,
            systems: editable.systems,
            measures: normalizedMeasures,
            measureNumberOverrideIDs: overrideIDs,
            restAnchorIDs: Set(normalizedRestCounts.keys),
            pickupAnchorIDs: normalizedPickupIDs,
            aiSuggestionMeasureIDs: normalizedAISuggestionIDs,
            ending1AnchorIDs: ending1IDs,
            ending2AnchorIDs: ending2IDs
        )
        renderSnapshot = snapshot
        currentJobID = snapshot.jobID
        currentRunID = snapshot.runID
        systems = snapshot.systems
        measures = snapshot.measures
        measureNumberOverrideValues = normalizedMeasureOverrides
        restMeasureCounts = normalizedRestCounts
        pickupMeasureIDs = normalizedPickupIDs
        manualRows = normalizedManualRows
        self.aiSuggestions = aiSuggestions
        self.aiSuggestRun = aiSuggestRun
        endingMeasureKinds = normalizedEndingKinds
        self.labelsMode = snapshot.labelsMode
        if normalizedMeasures.isEmpty {
            overlayGeometryWarning = "Measure geometry unavailable from backend."
        } else {
            overlayGeometryWarning = ""
        }
        syncAISuggestionReviewSelection()
    }

    private func normalizedManualRows(_ raw: [ManualRowState]?) -> [ManualRowState] {
        (raw ?? []).sorted { lhs, rhs in
            if lhs.page != rhs.page { return lhs.page < rhs.page }
            if lhs.rect.top != rhs.rect.top { return lhs.rect.top < rhs.rect.top }
            if lhs.rect.left != rhs.rect.left { return lhs.rect.left < rhs.rect.left }
            return lhs.manualRowId < rhs.manualRowId
        }
    }

    private func normalizedMeasureNumberOverrides(_ raw: [String: Int]?) -> [String: Int] {
        var cleaned: [String: Int] = [:]
        for (key, value) in raw ?? [:] {
            let measureID = key.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !measureID.isEmpty, value > 0 else { continue }
            cleaned[measureID] = value
        }
        return cleaned
    }

    private func normalizedRestMeasureCounts(_ raw: [String: Int]?) -> [String: Int] {
        var cleaned: [String: Int] = [:]
        for (key, value) in raw ?? [:] {
            let measureID = key.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !measureID.isEmpty, value > 0 else { continue }
            cleaned[measureID] = value
        }
        return cleaned
    }

    private func normalizedPickupMeasureIDs(_ raw: [String: Bool]?) -> Set<String> {
        var cleaned: Set<String> = []
        for (key, value) in raw ?? [:] {
            let measureID = key.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !measureID.isEmpty, value else { continue }
            cleaned.insert(measureID)
        }
        return cleaned
    }

    private func normalizedAISuggestionMeasureIDs(_ aiSuggestions: AISuggestionsState?, run: AISuggestRunState?) -> Set<String> {
        let status = run?.status?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? "idle"
        guard status == "completed" else { return [] }
        return Set((aiSuggestions?.by_measure_id ?? [:]).keys.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty })
    }

    private func normalizedEndingMeasureKinds(_ raw: [String: String]?) -> [String: EndingKind] {
        var cleaned: [String: EndingKind] = [:]
        for (key, value) in raw ?? [:] {
            let measureID = key.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedValue = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !measureID.isEmpty, let ending = EndingKind(rawValue: normalizedValue) else { continue }
            cleaned[measureID] = ending
        }
        return cleaned
    }

    private func orderedMeasureIndexByID() -> [String: Int] {
        let orderedMeasures = measures.sorted { lhs, rhs in
            if lhs.page != rhs.page { return lhs.page < rhs.page }
            let lhsSystemIndex = lhs.system_index ?? 0
            let rhsSystemIndex = rhs.system_index ?? 0
            if lhsSystemIndex != rhsSystemIndex { return lhsSystemIndex < rhsSystemIndex }
            if lhs.x_left != rhs.x_left { return lhs.x_left < rhs.x_left }
            let lhsLocalIndex = lhs.measure_local_index ?? 0
            let rhsLocalIndex = rhs.measure_local_index ?? 0
            if lhsLocalIndex != rhsLocalIndex { return lhsLocalIndex < rhsLocalIndex }
            return lhs.id < rhs.id
        }

        var indexByID: [String: Int] = [:]
        for (index, measure) in orderedMeasures.enumerated() {
            let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !measureID.isEmpty else { continue }
            indexByID[measureID] = index
        }
        return indexByID
    }

    private func orderedEndingIDs(
        _ ids: Set<String>,
        label: String,
        indexByID: [String: Int]
    ) throws -> [(id: String, index: Int)] {
        let indexed = ids.compactMap { measureID -> (id: String, index: Int)? in
            guard let index = indexByID[measureID] else { return nil }
            return (measureID, index)
        }
        guard indexed.count == ids.count else {
            throw LocalError("Could not locate every \(label) measure")
        }
        return indexed.sorted { lhs, rhs in lhs.index < rhs.index }
    }

    private func indicesAreContiguous(_ indices: [Int]) -> Bool {
        guard let first = indices.first else { return false }
        for (offset, index) in indices.enumerated() {
            if index != first + offset {
                return false
            }
        }
        return true
    }

    private func inferredSavedEndingGroups(
        indexByID: [String: Int]
    ) -> [EndingGroupSpan] {
        let savedRows = endingMeasureKinds.compactMap { measureID, kind -> (id: String, kind: EndingKind, index: Int)? in
            guard let index = indexByID[measureID] else { return nil }
            return (id: measureID, kind: kind, index: index)
        }
        .sorted { lhs, rhs in lhs.index < rhs.index }

        guard !savedRows.isEmpty else { return [] }

        var groups: [EndingGroupSpan] = []
        var currentIDs: [String] = []
        var currentStartIndex: Int?
        var currentEndIndex: Int?
        var previousIndex: Int?
        var previousKind: EndingKind?

        for row in savedRows {
            let startsNewGroup: Bool
            if let prevIndex = previousIndex, let prevKind = previousKind {
                startsNewGroup = row.index != prevIndex + 1 || (prevKind == .second && row.kind == .first)
            } else {
                startsNewGroup = false
            }

            if startsNewGroup,
               let currentStartValue = currentStartIndex,
               let currentEndValue = currentEndIndex,
               !currentIDs.isEmpty {
                groups.append(
                    EndingGroupSpan(
                        measureIDs: currentIDs,
                        indexRange: currentStartValue...currentEndValue
                    )
                )
                currentIDs = []
                currentStartIndex = nil
                currentEndIndex = nil
            }

            if currentStartIndex == nil {
                currentStartIndex = row.index
            }
            currentEndIndex = row.index
            currentIDs.append(row.id)
            previousIndex = row.index
            previousKind = row.kind
        }

        if let currentStartIndex,
           let currentEndIndex,
           !currentIDs.isEmpty {
            groups.append(
                EndingGroupSpan(
                    measureIDs: currentIDs,
                    indexRange: currentStartIndex...currentEndIndex
                )
            )
        }

        return groups
    }

    private func validatedGuidedEndingApplyPlan() throws -> GuidedEndingApplyPlan {
        guard !pendingEnding1MeasureIDs.isEmpty else {
            throw LocalError("Select at least one Ending 1 measure")
        }
        guard !pendingEnding2MeasureIDs.isEmpty else {
            throw LocalError("Select at least one Ending 2 measure")
        }

        let indexByID = orderedMeasureIndexByID()
        let ending1 = try orderedEndingIDs(pendingEnding1MeasureIDs, label: "Ending 1", indexByID: indexByID)
        let ending2 = try orderedEndingIDs(pendingEnding2MeasureIDs, label: "Ending 2", indexByID: indexByID)
        let ending1Indices = ending1.map { $0.index }
        let ending2Indices = ending2.map { $0.index }

        guard indicesAreContiguous(ending1Indices) else {
            throw LocalError("Ending 1 measures must be connected")
        }
        guard indicesAreContiguous(ending2Indices) else {
            throw LocalError("Ending 2 measures must be connected")
        }
        guard let ending1LastIndex = ending1Indices.last,
              let ending2FirstIndex = ending2Indices.first,
              ending1LastIndex + 1 == ending2FirstIndex else {
            throw LocalError("Ending 2 must start right after Ending 1")
        }

        let selectedMinIndex = min(ending1Indices[0], ending2Indices[0])
        let selectedMaxIndex = max(ending1Indices[ending1Indices.count - 1], ending2Indices[ending2Indices.count - 1])
        let selectedRange = selectedMinIndex...selectedMaxIndex
        let savedGroups = inferredSavedEndingGroups(indexByID: indexByID)
        if savedGroups.contains(where: { $0.indexRange.overlaps(selectedRange) }) {
            throw LocalError("This ending overlaps an existing ending. Remove it first.")
        }

        return GuidedEndingApplyPlan(
            ending1IDs: ending1.map { $0.id },
            ending2IDs: ending2.map { $0.id }
        )
    }

    private func toggleGuidedEndingSelection(measure: MeasureState) {
        guard !isBusy else { return }
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else {
            actionError = "Missing measure ID"
            return
        }

        switch guidedEndingSelectionPhase {
        case .selectingEnding1:
            if pendingEnding1MeasureIDs.contains(measureID) {
                pendingEnding1MeasureIDs.remove(measureID)
            } else {
                pendingEnding2MeasureIDs.remove(measureID)
                pendingEnding1MeasureIDs.insert(measureID)
            }
        case .selectingEnding2:
            if pendingEnding2MeasureIDs.contains(measureID) {
                pendingEnding2MeasureIDs.remove(measureID)
            } else {
                pendingEnding1MeasureIDs.remove(measureID)
                pendingEnding2MeasureIDs.insert(measureID)
            }
        }
    }

    private func applyPickup(measure: MeasureState) async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else {
            actionError = "Missing measure ID"
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Applying pickup..."
            let relabel = try await apiRelabel(jobID: jobID, edits: [
                RelabelEdit(type: "set_pickup_measure", system_id: nil, measure_id: measureID, intValue: nil, boolValue: true, stringValue: nil)
            ])
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )

            activeEditTool = .none
            phase = .ready
            detailNote = "Pickup applied"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func applyGuidedEndings() async {
        guard let jobID = currentJobID else {
            actionError = "No active job"
            return
        }
        let applyPlan: GuidedEndingApplyPlan
        do {
            applyPlan = try validatedGuidedEndingApplyPlan()
        } catch {
            actionError = error.localizedDescription
            return
        }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = "Applying endings..."
            let edits =
                applyPlan.ending1IDs.map { measureID in
                    RelabelEdit(type: "set_ending", system_id: nil, measure_id: measureID, intValue: nil, boolValue: nil, stringValue: EndingKind.first.rawValue)
                } +
                applyPlan.ending2IDs.map { measureID in
                    RelabelEdit(type: "set_ending", system_id: nil, measure_id: measureID, intValue: nil, boolValue: nil, stringValue: EndingKind.second.rawValue)
                }
            let relabel = try await apiRelabel(jobID: jobID, edits: edits)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: labelsModeFromState(state),
                token: token,
                preserveViewport: true
            )

            resetGuidedEndingDraft()
            activeEditTool = .none
            phase = .ready
            detailNote = "Endings applied"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func savedMeasureNumberOverride(for measure: MeasureState) -> Int? {
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else { return nil }
        return measureNumberOverrideValues[measureID]
    }

    private func savedRestCount(for measure: MeasureState) -> Int? {
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else { return nil }
        return restMeasureCounts[measureID]
    }

    private func savedPickup(for measure: MeasureState) -> Bool {
        let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !measureID.isEmpty else { return false }
        return pickupMeasureIDs.contains(measureID)
    }

    private func hasSavedEdits(for measure: MeasureState) -> Bool {
        guard measure.excluded_from_counting != true else { return false }
        return savedMeasureNumberOverride(for: measure) != nil || savedRestCount(for: measure) != nil || savedPickup(for: measure)
    }

    private func savedEdits(for measure: MeasureState) -> [SavedMeasureEdit] {
        let savedMeasureOverride = savedMeasureNumberOverride(for: measure)
        let hasSavedPickup = savedPickup(for: measure)
        let savedRest = savedRestCount(for: measure)

        return SavedMeasureEditKind.displayOrder.compactMap { kind in
            switch kind {
            case .measureNumber:
                return savedMeasureOverride.map(SavedMeasureEdit.measureNumber)
            case .pickup:
                return hasSavedPickup ? .pickup : nil
            case .rest:
                return savedRest.map(SavedMeasureEdit.rest)
            }
        }
    }

    private func measureEditsSheetHeight(for measure: MeasureState) -> CGFloat {
        switch savedEdits(for: measure).count {
        case 0, 1:
            return 270
        case 2:
            return 320
        default:
            return 370
        }
    }

    private var measureNumberSheetDetents: Set<PresentationDetent> {
        if isCompactWidth {
            return [.medium, .large]
        }
        return [.height(260)]
    }

    private var restSheetDetents: Set<PresentationDetent> {
        if isCompactWidth {
            return [.medium, .large]
        }
        return [.height(300)]
    }

    private func measureEditsSheetDetents(for measure: MeasureState) -> Set<PresentationDetent> {
        if isCompactWidth {
            return savedEdits(for: measure).count > 1 ? [.medium, .large] : [.medium]
        }
        return [.height(measureEditsSheetHeight(for: measure))]
    }

    private func pdfCardHeight(for size: CGSize) -> CGFloat {
        if isCompactWidth {
            return max(300, min(420, size.height * 0.50))
        }
        return max(380, size.height * 0.66)
    }

    private var connectionStatusDot: some View {
        Circle()
            .fill(isServerReachable == true ? Color.green : isServerReachable == false ? Color.red : Color.secondary)
            .frame(width: 8, height: 8)
    }

    private var connectionStatusColor: Color {
        isServerReachable == true ? .green : isServerReachable == false ? .red : .secondary
    }

    private var bannerStatusRow: some View {
        HStack(spacing: 8) {
            connectionStatusDot

            Text(connectionStatusText)
                .font(.caption.weight(.medium))
                .foregroundStyle(connectionStatusColor)
                .lineLimit(1)

            if phase != .idle {
                Text("·")
                    .foregroundStyle(.secondary)
                Text(phaseLabel)
                    .font(.caption)
                    .foregroundStyle(phaseColor)
                    .lineLimit(1)
            }
        }
    }

    private var connectionBannerButtons: some View {
        HStack(spacing: 8) {
            if isServerReachable == false {
                Button {
                    Task { await checkServerConnectivity() }
                } label: {
                    Text("Retry")
                        .font(.caption.weight(.semibold))
                }
                .buttonStyle(.bordered)
                .controlSize(.mini)
            }

            Button {
                pickPDF()
            } label: {
                Label("Upload PDF", systemImage: "doc.badge.plus")
                    .labelStyle(.titleAndIcon)
                    .font(.caption.weight(.semibold))
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.mini)
            .disabled(isBusy)
        }
    }

    private func applyLabelsMode(_ mode: LabelsMode) async {
        guard let jobID = currentJobID else { return }
        guard !isBusy else { return }
        let token = activeJobToken

        isBusy = true
        defer { isBusy = false }

        do {
            detailNote = mode == .allMeasures ? "Applying all-measure labels..." : "Applying staff-start labels..."
            let relabel = try await apiRelabel(jobID: jobID, edits: [
                RelabelEdit(type: "set_labels_mode", system_id: nil, measure_id: nil, intValue: nil, boolValue: nil, stringValue: mode.rawValue)
            ])
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            try validateRelabelOutcome(relabel)

            let newURL = relabel.artifacts_http?.audiveris_out_corrected_pdf?.nonEmpty
                ?? relabel.artifacts_http?.audiveris_out_pdf?.nonEmpty
                ?? correctedPDFURL
                ?? baselinePDFURL

            guard let finalURL = newURL else {
                throw LocalError("Rendered PDF not ready")
            }

            let pdfData = try await downloadPDF(urlString: finalURL)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            correctedPDFURL = relabel.artifacts_http?.audiveris_out_corrected_pdf ?? correctedPDFURL

            let state = try await apiFetchState(jobID: jobID)
            guard isTokenCurrent(token, expectedJobID: jobID) else { return }
            let effectiveMode = labelsModeFromState(state)
            commitRenderSnapshot(
                jobID: jobID,
                runID: currentRunID,
                pdfData: pdfData,
                editable: state.editable_state,
                aiSuggestions: state.ai_suggestions,
                aiSuggestRun: state.ai_suggest_run,
                labelsMode: effectiveMode,
                token: token
            )

            phase = .ready
            detailNote = mode == .allMeasures ? "All-measure labels ready" : "Staff-start labels ready"
        } catch is CancellationError {
            return
        } catch {
            phase = .failed
            detailNote = error.localizedDescription
            actionError = error.localizedDescription
        }
    }

    private func downloadPDF(urlString: String) async throws -> Data {
        guard let url = URL(string: urlString) else {
            throw LocalError("Invalid PDF URL")
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw LocalError("Failed to download rendered PDF")
        }
        return data
    }

    private func validateHTTP(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw LocalError("Invalid response")
        }
        guard (200...299).contains(http.statusCode) else {
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let error = obj["error"] {
                if let message = error as? String {
                    throw LocalError(message)
                }
                if let errorDict = error as? [String: Any],
                   let message = errorDict["message"] as? String {
                    throw LocalError(message)
                }
            }
            if let text = String(data: data, encoding: .utf8), !text.isEmpty {
                throw LocalError(text)
            }
            throw LocalError("HTTP \(http.statusCode)")
        }
    }

    private func clearRuntimeForNewRun() {
        phase = .idle
        detailNote = ""
        currentJobID = nil
        currentRunID = nil
        baselinePDFURL = nil
        correctedPDFURL = nil
        renderSnapshot = nil
        drawnOverlayCount = 0
        systems = []
        measures = []
        labelsMode = .allMeasures
        measureNumberOverrideValues = [:]
        restMeasureCounts = [:]
        pickupMeasureIDs = []
        manualRows = []
        manualDraftRows = []
        autoDraftRows = []
        manualDraftPage = nil
        manualFixTool = .addRow
        manualStaffKind = .single
        manualSelection = nil
        autoSelection = nil
        pendingManualFixDelete = nil
        pendingLabelEraseArea = nil
        currentVisiblePDFPage = 1
        aiSuggestions = nil
        aiSuggestRun = nil
        isReviewingAISuggestions = false
        currentAISuggestionMeasureID = nil
        isGeneratingAISuggestions = false
        endingMeasureKinds = [:]
        guidedEndingSelectionPhase = .selectingEnding1
        pendingEnding1MeasureIDs = []
        pendingEnding2MeasureIDs = []
        activeEditTool = .none
        pendingAutoScrollToTools = false
        pendingAutoScrollToPDF = false
        overlayGeometryWarning = ""
        activeEditSheet = nil
        measureEditValue = ""
        restEditValue = ""
        actionError = nil
    }

    private func clearSession() {
        activeJobToken = UUID()
        persistedJobID = ""
        persistedPDFName = ""
        persistedJobSavedAt = 0
        currentJobID = nil
        currentRunID = nil
        baselinePDFURL = nil
        correctedPDFURL = nil
        renderSnapshot = nil
        drawnOverlayCount = 0
        systems = []
        measures = []
        labelsMode = .allMeasures
        measureNumberOverrideValues = [:]
        restMeasureCounts = [:]
        pickupMeasureIDs = []
        manualRows = []
        manualDraftRows = []
        autoDraftRows = []
        manualDraftPage = nil
        manualFixTool = .addRow
        manualStaffKind = .single
        manualSelection = nil
        autoSelection = nil
        pendingManualFixDelete = nil
        pendingLabelEraseArea = nil
        currentVisiblePDFPage = 1
        aiSuggestions = nil
        aiSuggestRun = nil
        isReviewingAISuggestions = false
        currentAISuggestionMeasureID = nil
        isGeneratingAISuggestions = false
        endingMeasureKinds = [:]
        guidedEndingSelectionPhase = .selectingEnding1
        pendingEnding1MeasureIDs = []
        pendingEnding2MeasureIDs = []
        activeEditTool = .none
        pendingAutoScrollToTools = false
        pendingAutoScrollToPDF = false
        overlayGeometryWarning = ""
        activeEditSheet = nil
        measureEditValue = ""
        restEditValue = ""
        phase = .idle
        detailNote = ""
    }
}

// MARK: - Edit Sheet

private enum SavedMeasureEditKind: CaseIterable {
    case measureNumber
    case pickup
    case rest

    static let displayOrder: [SavedMeasureEditKind] = [
        .measureNumber,
        .pickup,
        .rest
    ]
}

private enum SavedMeasureEdit: Identifiable, Hashable {
    case measureNumber(Int)
    case pickup
    case rest(Int)

    var id: String {
        switch self {
        case .measureNumber:
            return "measure-number"
        case .pickup:
            return "pickup"
        case .rest:
            return "rest"
        }
    }

    var title: String {
        switch self {
        case .measureNumber:
            return "Measure #"
        case .pickup:
            return "Pickup"
        case .rest:
            return "Rest"
        }
    }

    var valueText: String {
        switch self {
        case .measureNumber(let value), .rest(let value):
            return String(value)
        case .pickup:
            return "Saved"
        }
    }
}

private struct MeasureEditsSheet: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    let measure: MeasureState
    let savedEdits: [SavedMeasureEdit]
    let isBusy: Bool
    let onChangeMeasureNumber: () -> Void
    let onRemoveMeasureNumber: () -> Void
    let onChangeRest: () -> Void
    let onRemoveRest: () -> Void
    let onRemovePickup: () -> Void

    private var isCompactWidth: Bool { horizontalSizeClass == .compact }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Measure Edits")
                        .font(.title3.weight(.semibold))

                    Text("Current measure number: \(measure.current_value ?? "—")")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    if savedEdits.isEmpty {
                        Text("No saved edits on this measure.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(Array(savedEdits.enumerated()), id: \.element.id) { index, edit in
                            VStack(alignment: .leading, spacing: 10) {
                                if isCompactWidth {
                                    VStack(alignment: .leading, spacing: 10) {
                                        editMeta(edit)
                                        editButtons(edit)
                                    }
                                } else {
                                    HStack(alignment: .top, spacing: 12) {
                                        editMeta(edit)
                                        Spacer()
                                        editButtons(edit)
                                    }
                                }

                                if index < savedEdits.count - 1 {
                                    Divider()
                                }
                            }
                        }
                    }

                    Spacer(minLength: 0)
                }
                .padding(16)
            }
            .scrollDismissesKeyboard(.interactively)
        }
    }

    @ViewBuilder
    private func editMeta(_ edit: SavedMeasureEdit) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(edit.title)
                .font(.subheadline.weight(.semibold))
            Text(edit.valueText)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func editButtons(_ edit: SavedMeasureEdit) -> some View {
        if isCompactWidth {
            VStack(spacing: 8) {
                if edit != .pickup {
                    changeButton(for: edit)
                }
                removeButton(for: edit)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            HStack(spacing: 8) {
                if edit != .pickup {
                    changeButton(for: edit)
                }
                removeButton(for: edit)
            }
        }
    }

    private func changeButton(for edit: SavedMeasureEdit) -> some View {
        Button("Change") {
            switch edit {
            case .measureNumber:
                onChangeMeasureNumber()
            case .rest:
                onChangeRest()
            case .pickup:
                return
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(isBusy)
    }

    private func removeButton(for edit: SavedMeasureEdit) -> some View {
        Button("Remove", role: .destructive) {
            switch edit {
            case .measureNumber:
                onRemoveMeasureNumber()
            case .pickup:
                onRemovePickup()
            case .rest:
                onRemoveRest()
            }
        }
        .buttonStyle(.bordered)
        .controlSize(.small)
        .disabled(isBusy)
    }
}

private struct EditMeasureSheet: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    let measure: MeasureState
    @Binding var draftValue: String
    let isBusy: Bool
    let onApply: () -> Void

    private var isCompactWidth: Bool { horizontalSizeClass == .compact }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Edit Measure Number")
                        .font(.title3.weight(.semibold))

                    Text("Current number: \(measure.current_value ?? "—")")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Enter new measure number")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        TextField("New number", text: $draftValue)
                            .keyboardType(.numberPad)
                            .textFieldStyle(.roundedBorder)
                    }

                    Button {
                        onApply()
                    } label: {
                        if isBusy {
                            ProgressView()
                                .frame(maxWidth: .infinity)
                        } else {
                            Text("Apply")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isBusy)

                    Spacer(minLength: isCompactWidth ? 48 : 0)
                }
                .padding(16)
            }
            .scrollDismissesKeyboard(.interactively)
        }
    }
}

private struct EditRestSheet: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    let measure: MeasureState
    let currentRestCount: Int?
    @Binding var draftValue: String
    let isBusy: Bool
    let onApply: () -> Void

    private var isCompactWidth: Bool { horizontalSizeClass == .compact }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Edit Rest")
                        .font(.title3.weight(.semibold))

                    Text("Current measure number: \(measure.current_value ?? "—")")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    Text("Saved rest count: \(currentRestCount.map(String.init) ?? "None")")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Enter rest count")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                        TextField("Rest count", text: $draftValue)
                            .keyboardType(.numberPad)
                            .textFieldStyle(.roundedBorder)
                    }

                    Button {
                        onApply()
                    } label: {
                        if isBusy {
                            ProgressView()
                                .frame(maxWidth: .infinity)
                        } else {
                            Text("Apply")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isBusy)

                    Spacer(minLength: isCompactWidth ? 48 : 0)
                }
                .padding(16)
            }
            .scrollDismissesKeyboard(.interactively)
        }
    }
}

// MARK: - PDF Overlay View

private struct PDFOverlayContainer: UIViewRepresentable {
    let pdfData: Data
    let snapshotToken: UUID
    let documentLoadID: UUID
    let preserveViewport: Bool
    let systems: [SystemState]
    let measures: [MeasureState]
    let measureNumberOverrideIDs: Set<String>
    let restAnchorIDs: Set<String>
    let pickupAnchorIDs: Set<String>
    let aiSuggestionMeasureIDs: Set<String>
    let ending1AnchorIDs: Set<String>
    let ending2AnchorIDs: Set<String>
    let pendingEnding1IDs: Set<String>
    let pendingEnding2IDs: Set<String>
    let labelsMode: LabelsMode
    let manualEditor: ManualEditorState?
    let autoEditor: AutoEditorState?
    let pendingLabelEraseArea: LabelEraseAreaState?
    let onOverlayCount: (Int) -> Void
    let onVisiblePageChange: (Int) -> Void
    let onManualRowsChange: ([ManualRowState]) -> Void
    let onManualSelectionChange: (ManualSelectionState?) -> Void
    let onAutoRowsChange: ([AutoRowState]) -> Void
    let onAutoSelectionChange: (AutoSelectionState?) -> Void
    let onLabelEraseAreaChange: (LabelEraseAreaState?) -> Void
    let onSelectMeasure: (MeasureState) -> Void

    func makeUIView(context: Context) -> OverlayPDFView {
        let view = OverlayPDFView()
        view.onSelectMeasure = onSelectMeasure
        view.onOverlayCount = onOverlayCount
        view.onVisiblePageChange = onVisiblePageChange
        view.onManualRowsChange = onManualRowsChange
        view.onManualSelectionChange = onManualSelectionChange
        view.onAutoRowsChange = onAutoRowsChange
        view.onAutoSelectionChange = onAutoSelectionChange
        view.onLabelEraseAreaChange = onLabelEraseAreaChange
        return view
    }

    func updateUIView(_ uiView: OverlayPDFView, context: Context) {
        uiView.onSelectMeasure = onSelectMeasure
        uiView.onOverlayCount = onOverlayCount
        uiView.onVisiblePageChange = onVisiblePageChange
        uiView.onManualRowsChange = onManualRowsChange
        uiView.onManualSelectionChange = onManualSelectionChange
        uiView.onAutoRowsChange = onAutoRowsChange
        uiView.onAutoSelectionChange = onAutoSelectionChange
        uiView.onLabelEraseAreaChange = onLabelEraseAreaChange
        uiView.update(
            pdfData: pdfData,
            snapshotToken: snapshotToken,
            documentLoadID: documentLoadID,
            preserveViewport: preserveViewport,
            systems: systems,
            measures: measures,
            measureNumberOverrideIDs: measureNumberOverrideIDs,
            restAnchorIDs: restAnchorIDs,
            pickupAnchorIDs: pickupAnchorIDs,
            aiSuggestionMeasureIDs: aiSuggestionMeasureIDs,
            ending1AnchorIDs: ending1AnchorIDs,
            ending2AnchorIDs: ending2AnchorIDs,
            pendingEnding1IDs: pendingEnding1IDs,
            pendingEnding2IDs: pendingEnding2IDs,
            labelsMode: labelsMode,
            manualEditor: manualEditor,
            autoEditor: autoEditor,
            pendingLabelEraseArea: pendingLabelEraseArea
        )
    }
}

private final class OverlayPDFView: UIView, UIGestureRecognizerDelegate {
    private struct OverlayRow {
        let rowID: String
        let measureID: String
        let systemID: String
        let page: Int
        let pageRect: CGRect
    }

    private enum ManualCorner: String {
        case topLeft
        case topRight
        case bottomLeft
        case bottomRight
    }

    private enum ManualDragState {
        case addRow(page: Int, start: CGPoint, current: CGPoint)
        case addCut(rowID: String, currentX: Double)
        case resize(rowID: String, corner: ManualCorner, startRect: ManualRowRect, currentPoint: CGPoint)
        case addAutoCut(rowID: String, boxIndex: Int, currentX: Double)
        case resizeAuto(rowID: String, corner: ManualCorner, startRect: ManualRowRect, currentPoint: CGPoint)
    }

    private struct SavedViewport {
        let pageIndex: Int
        let scaleFactor: CGFloat
        let normalizedCenter: CGPoint
        let normalizedVisibleRect: CGRect
    }

    private struct PendingViewportRestore {
        let documentLoadID: UUID
        let viewport: SavedViewport
    }

    private let pdfView = PDFView()
    private let overlayLayer = CALayer()
    private let manualLayer = CALayer()
    private var currentSnapshotToken: UUID = UUID()
    private var currentDocumentLoadID: UUID = UUID()
    private var currentMeasures: [MeasureState] = []
    private var currentMeasureNumberOverrideIDs: Set<String> = []
    private var currentRestAnchorIDs: Set<String> = []
    private var currentPickupAnchorIDs: Set<String> = []
    private var currentAISuggestionMeasureIDs: Set<String> = []
    private var currentEnding1AnchorIDs: Set<String> = []
    private var currentEnding2AnchorIDs: Set<String> = []
    private var currentPendingEnding1IDs: Set<String> = []
    private var currentPendingEnding2IDs: Set<String> = []
    private var currentLabelsMode: LabelsMode = .allMeasures
    private var currentManualEditor: ManualEditorState?
    private var currentAutoEditor: AutoEditorState?
    private var currentPendingLabelEraseArea: LabelEraseAreaState?
    private var manualDragState: ManualDragState?
    private var lastOverlayLogSignature: String = ""
    private var pendingViewportRestore: PendingViewportRestore?
    private var panGesture: UIPanGestureRecognizer?
    private var manualInteractionLocked = false

    var onSelectMeasure: ((MeasureState) -> Void)?
    var onOverlayCount: ((Int) -> Void)?
    var onVisiblePageChange: ((Int) -> Void)?
    var onManualRowsChange: (([ManualRowState]) -> Void)?
    var onManualSelectionChange: ((ManualSelectionState?) -> Void)?
    var onAutoRowsChange: (([AutoRowState]) -> Void)?
    var onAutoSelectionChange: ((AutoSelectionState?) -> Void)?
    var onLabelEraseAreaChange: ((LabelEraseAreaState?) -> Void)?

    override init(frame: CGRect) {
        super.init(frame: frame)
        setup()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setup()
    }

    private func setup() {
        backgroundColor = .clear

        pdfView.displayMode = .singlePageContinuous
        pdfView.displayDirection = .vertical
        pdfView.autoScales = true
        pdfView.minScaleFactor = 0.5
        pdfView.maxScaleFactor = 5.0
        pdfView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(pdfView)

        NSLayoutConstraint.activate([
            pdfView.leadingAnchor.constraint(equalTo: leadingAnchor),
            pdfView.trailingAnchor.constraint(equalTo: trailingAnchor),
            pdfView.topAnchor.constraint(equalTo: topAnchor),
            pdfView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])

        overlayLayer.masksToBounds = false
        overlayLayer.name = "system-overlays"
        manualLayer.masksToBounds = false
        manualLayer.name = "manual-overlays"

        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        tap.cancelsTouchesInView = false
        tap.delaysTouchesBegan = false
        tap.delaysTouchesEnded = false
        tap.delegate = self
        addGestureRecognizer(tap)

        let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        pan.cancelsTouchesInView = false
        pan.delegate = self
        addGestureRecognizer(pan)
        panGesture = pan

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(redrawOverlays),
            name: Notification.Name.PDFViewScaleChanged,
            object: pdfView
        )

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(redrawOverlays),
            name: Notification.Name.PDFViewPageChanged,
            object: pdfView
        )

        suppressPDFTextInteractions()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }

    func update(
        pdfData: Data,
        snapshotToken: UUID,
        documentLoadID: UUID,
        preserveViewport: Bool,
        systems: [SystemState],
        measures: [MeasureState],
        measureNumberOverrideIDs: Set<String>,
        restAnchorIDs: Set<String>,
        pickupAnchorIDs: Set<String>,
            aiSuggestionMeasureIDs: Set<String>,
            ending1AnchorIDs: Set<String>,
            ending2AnchorIDs: Set<String>,
            pendingEnding1IDs: Set<String>,
            pendingEnding2IDs: Set<String>,
            labelsMode: LabelsMode,
            manualEditor: ManualEditorState?,
            autoEditor: AutoEditorState?,
            pendingLabelEraseArea: LabelEraseAreaState?
    ) {
        currentSnapshotToken = snapshotToken
        if documentLoadID != currentDocumentLoadID {
            let savedViewport = preserveViewport ? captureCurrentViewport() : nil
            currentDocumentLoadID = documentLoadID
            pendingViewportRestore = nil
            if let doc = PDFDocument(data: pdfData) {
                pdfView.document = doc
                suppressPDFTextInteractions()
                if let savedViewport {
                    pendingViewportRestore = PendingViewportRestore(documentLoadID: documentLoadID, viewport: savedViewport)
                    restoreViewport(savedViewport, documentLoadID: documentLoadID, scheduleSecondPass: true)
                } else {
                    resetToDefaultFit(documentLoadID: documentLoadID, scheduleSecondPass: true)
                }
            }
        }
        currentMeasures = measures
        currentMeasureNumberOverrideIDs = measureNumberOverrideIDs
        currentRestAnchorIDs = restAnchorIDs
        currentPickupAnchorIDs = pickupAnchorIDs
        currentAISuggestionMeasureIDs = aiSuggestionMeasureIDs
        currentEnding1AnchorIDs = ending1AnchorIDs
        currentEnding2AnchorIDs = ending2AnchorIDs
        currentPendingEnding1IDs = pendingEnding1IDs
        currentPendingEnding2IDs = pendingEnding2IDs
        currentLabelsMode = labelsMode
        currentManualEditor = manualEditor
        currentAutoEditor = autoEditor
        currentPendingLabelEraseArea = pendingLabelEraseArea
        suppressPDFTextInteractions()
        setManualInteractionLocked(manualEditor != nil || autoEditor != nil)
        if let manualEditor,
           let selection = manualEditor.selection,
           !manualEditor.rows.contains(where: { $0.manualRowId == selection.rowID }) {
            onManualSelectionChange?(nil)
        }
        if let autoEditor,
           let selection = autoEditor.selection,
           !autoEditor.rows.contains(where: { $0.systemID == selection.rowID }) {
            onAutoSelectionChange?(nil)
        }
        notifyVisiblePageIfNeeded()
        redrawOverlays()
    }

    private func captureCurrentViewport() -> SavedViewport? {
        guard let doc = pdfView.document,
              doc.pageCount > 0,
              let scrollView = embeddedScrollView(),
              let documentView = pdfView.documentView else { return nil }

        let visibleRect = visibleRectInDocumentView(scrollView: scrollView)
        let visibleCenter = visibleCenterInDocumentView(scrollView: scrollView)
        guard let page = pageContainingDocumentPoint(visibleCenter, in: doc, documentView: documentView) ?? pdfView.currentPage else {
            return nil
        }

        var pageIndex: Int?
        for index in 0..<doc.pageCount {
            if doc.page(at: index) === page {
                pageIndex = index
                break
            }
        }
        guard let pageIndex else { return nil }

        let pageBounds = page.bounds(for: .mediaBox)
        guard pageBounds.width > 0, pageBounds.height > 0 else { return nil }

        guard let visibleRectOnPage = documentRectToPageRect(visibleRect, page: page, documentView: documentView) else {
            return nil
        }

        let normalizedVisibleRect = CGRect(
            x: (visibleRectOnPage.minX - pageBounds.minX) / pageBounds.width,
            y: (visibleRectOnPage.minY - pageBounds.minY) / pageBounds.height,
            width: visibleRectOnPage.width / pageBounds.width,
            height: visibleRectOnPage.height / pageBounds.height
        )

        let normalizedX = (visibleRectOnPage.midX - pageBounds.minX) / pageBounds.width
        let normalizedY = (visibleRectOnPage.midY - pageBounds.minY) / pageBounds.height

        return SavedViewport(
            pageIndex: pageIndex,
            scaleFactor: pdfView.scaleFactor,
            normalizedCenter: CGPoint(x: normalizedX, y: normalizedY),
            normalizedVisibleRect: normalizedVisibleRect
        )
    }

    private func visibleRectInDocumentView(scrollView: UIScrollView) -> CGRect {
        CGRect(
            x: scrollView.contentOffset.x + scrollView.adjustedContentInset.left,
            y: scrollView.contentOffset.y + scrollView.adjustedContentInset.top,
            width: scrollView.bounds.width,
            height: scrollView.bounds.height
        )
    }

    private func visibleCenterInDocumentView(scrollView: UIScrollView) -> CGPoint {
        let rect = visibleRectInDocumentView(scrollView: scrollView)
        return CGPoint(x: rect.midX, y: rect.midY)
    }

    private func pageContainingDocumentPoint(_ point: CGPoint, in document: PDFDocument, documentView: UIView) -> PDFPage? {
        for index in 0..<document.pageCount {
            guard let page = document.page(at: index) else { continue }
            let rectInPDFView = pdfView.convert(page.bounds(for: .mediaBox), from: page)
            let rect = documentView.convert(rectInPDFView, from: pdfView)
            if rect.contains(point) {
                return page
            }
        }
        return nil
    }

    private func documentRectToPageRect(_ rect: CGRect, page: PDFPage, documentView: UIView) -> CGRect? {
        guard rect.width > 0, rect.height > 0 else { return nil }

        let topLeftInPDFView = pdfView.convert(rect.origin, from: documentView)
        let bottomRightInPDFView = pdfView.convert(CGPoint(x: rect.maxX, y: rect.maxY), from: documentView)
        let point1 = pdfView.convert(topLeftInPDFView, to: page)
        let point2 = pdfView.convert(bottomRightInPDFView, to: page)

        let pageRect = CGRect(
            x: min(point1.x, point2.x),
            y: min(point1.y, point2.y),
            width: abs(point2.x - point1.x),
            height: abs(point2.y - point1.y)
        )
        guard pageRect.width.isFinite, pageRect.height.isFinite else { return nil }
        return pageRect
    }

    private func pageRectToDocumentRect(_ rect: CGRect, page: PDFPage, documentView: UIView) -> CGRect? {
        guard rect.width.isFinite, rect.height.isFinite else { return nil }

        let topLeftInPDFView = pdfView.convert(CGPoint(x: rect.minX, y: rect.minY), from: page)
        let bottomRightInPDFView = pdfView.convert(CGPoint(x: rect.maxX, y: rect.maxY), from: page)
        let point1 = documentView.convert(topLeftInPDFView, from: pdfView)
        let point2 = documentView.convert(bottomRightInPDFView, from: pdfView)

        let documentRect = CGRect(
            x: min(point1.x, point2.x),
            y: min(point1.y, point2.y),
            width: abs(point2.x - point1.x),
            height: abs(point2.y - point1.y)
        )
        guard documentRect.width.isFinite, documentRect.height.isFinite else { return nil }
        return documentRect
    }

    private func restoreViewport(_ viewport: SavedViewport, documentLoadID: UUID, scheduleSecondPass: Bool) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            guard self.currentDocumentLoadID == documentLoadID else { return }
            guard let document = self.pdfView.document, document.pageCount > 0 else { return }

            let pageIndex = min(max(viewport.pageIndex, 0), max(document.pageCount - 1, 0))
            guard let page = document.page(at: pageIndex) else { return }

            self.applySavedScaleFactor(viewport.scaleFactor)
            self.layoutIfNeeded()
            self.pdfView.layoutIfNeeded()
            self.pdfView.documentView?.layoutIfNeeded()

            let pageBounds = page.bounds(for: .mediaBox)
            guard let documentView = self.pdfView.documentView,
                  let scrollView = self.embeddedScrollView() else {
                self.pdfView.go(to: page)
                self.redrawOverlays()
                return
            }

            self.applySavedViewport(viewport, page: page, pageBounds: pageBounds, documentView: documentView, scrollView: scrollView)

            if scheduleSecondPass {
                DispatchQueue.main.async { [weak self] in
                    self?.performPendingViewportRestore(for: documentLoadID)
                }
            } else {
                self.pendingViewportRestore = nil
            }
            self.redrawOverlays()
        }
    }

    private func resetToDefaultFit(documentLoadID: UUID, scheduleSecondPass: Bool) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            guard self.currentDocumentLoadID == documentLoadID else { return }
            guard let document = self.pdfView.document, document.pageCount > 0 else { return }

            self.pdfView.autoScales = true
            self.layoutIfNeeded()
            self.pdfView.layoutIfNeeded()
            self.pdfView.documentView?.layoutIfNeeded()

            if let firstPage = document.page(at: 0) {
                self.pdfView.go(to: firstPage)
            }

            if let scrollView = self.embeddedScrollView() {
                let topOffset = CGPoint(
                    x: -scrollView.adjustedContentInset.left,
                    y: -scrollView.adjustedContentInset.top
                )
                scrollView.setContentOffset(topOffset, animated: false)
            }

            if scheduleSecondPass {
                DispatchQueue.main.async { [weak self] in
                    self?.resetToDefaultFit(documentLoadID: documentLoadID, scheduleSecondPass: false)
                }
            }

            self.redrawOverlays()
        }
    }

    private func performPendingViewportRestore(for documentLoadID: UUID) {
        guard let pending = pendingViewportRestore,
              pending.documentLoadID == documentLoadID,
              currentDocumentLoadID == documentLoadID,
              let document = pdfView.document,
              document.pageCount > 0 else {
            return
        }

        let pageIndex = min(max(pending.viewport.pageIndex, 0), max(document.pageCount - 1, 0))
        guard let page = document.page(at: pageIndex),
              let documentView = pdfView.documentView,
              let scrollView = embeddedScrollView() else {
            pendingViewportRestore = nil
            return
        }

        applySavedScaleFactor(pending.viewport.scaleFactor)
        layoutIfNeeded()
        pdfView.layoutIfNeeded()
        documentView.layoutIfNeeded()

        let pageBounds = page.bounds(for: .mediaBox)
        applySavedViewport(pending.viewport, page: page, pageBounds: pageBounds, documentView: documentView, scrollView: scrollView)
        pendingViewportRestore = nil
        redrawOverlays()
    }

    private func applySavedScaleFactor(_ factor: CGFloat) {
        let clampedScale = min(max(factor, pdfView.minScaleFactor), pdfView.maxScaleFactor)
        pdfView.scaleFactor = clampedScale
    }

    private func applySavedViewport(
        _ viewport: SavedViewport,
        page: PDFPage,
        pageBounds: CGRect,
        documentView: UIView,
        scrollView: UIScrollView
    ) {
        let visibleRectOnPage = CGRect(
            x: pageBounds.minX + (pageBounds.width * viewport.normalizedVisibleRect.minX),
            y: pageBounds.minY + (pageBounds.height * viewport.normalizedVisibleRect.minY),
            width: pageBounds.width * viewport.normalizedVisibleRect.width,
            height: pageBounds.height * viewport.normalizedVisibleRect.height
        )

        if let targetDocumentRect = pageRectToDocumentRect(visibleRectOnPage, page: page, documentView: documentView),
           targetDocumentRect.width > 0,
           targetDocumentRect.height > 0 {
            let offset = clampedContentOffset(
                x: targetDocumentRect.minX - scrollView.adjustedContentInset.left,
                y: targetDocumentRect.minY - scrollView.adjustedContentInset.top,
                scrollView: scrollView
            )
            scrollView.setContentOffset(offset, animated: false)
            return
        }

        let pointOnPage = CGPoint(
            x: pageBounds.minX + (pageBounds.width * viewport.normalizedCenter.x),
            y: pageBounds.minY + (pageBounds.height * viewport.normalizedCenter.y)
        )
        let pointInPDFView = pdfView.convert(pointOnPage, from: page)
        let pointInDocument = documentView.convert(pointInPDFView, from: pdfView)
        let fallbackOffset = clampedContentOffset(
            x: pointInDocument.x - (scrollView.bounds.width / 2) - scrollView.adjustedContentInset.left,
            y: pointInDocument.y - (scrollView.bounds.height / 2) - scrollView.adjustedContentInset.top,
            scrollView: scrollView
        )
        scrollView.setContentOffset(fallbackOffset, animated: false)
    }

    private func clampedContentOffset(x: CGFloat, y: CGFloat, scrollView: UIScrollView) -> CGPoint {
        let minOffsetX = -scrollView.adjustedContentInset.left
        let minOffsetY = -scrollView.adjustedContentInset.top
        let maxOffsetX = max(minOffsetX, scrollView.contentSize.width - scrollView.bounds.width + scrollView.adjustedContentInset.right)
        let maxOffsetY = max(minOffsetY, scrollView.contentSize.height - scrollView.bounds.height + scrollView.adjustedContentInset.bottom)
        return CGPoint(
            x: min(max(x, minOffsetX), maxOffsetX),
            y: min(max(y, minOffsetY), maxOffsetY)
        )
    }

    private func embeddedScrollView() -> UIScrollView? {
        if let direct = pdfView.subviews.first(where: { $0 is UIScrollView }) as? UIScrollView {
            return direct
        }
        for subview in pdfView.subviews {
            if let nested = subview.subviews.first(where: { $0 is UIScrollView }) as? UIScrollView {
                return nested
            }
        }
        return nil
    }

    private func suppressPDFTextInteractions() {
        disableLongPressGestures(in: pdfView)
        if let documentView = pdfView.documentView {
            disableLongPressGestures(in: documentView)
        }
    }

    private func disableLongPressGestures(in view: UIView) {
        view.gestureRecognizers?.forEach { gesture in
            if gesture is UILongPressGestureRecognizer {
                gesture.isEnabled = false
            }
        }
        for subview in view.subviews {
            disableLongPressGestures(in: subview)
        }
    }

    private func setManualInteractionLocked(_ locked: Bool) {
        guard manualInteractionLocked != locked else { return }
        manualInteractionLocked = locked
        if let scrollView = embeddedScrollView() {
            scrollView.isScrollEnabled = !locked
            scrollView.pinchGestureRecognizer?.isEnabled = !locked
        }
    }

    private func notifyVisiblePageIfNeeded() {
        guard let document = pdfView.document else { return }
        guard let currentPage = pdfView.currentPage ?? document.page(at: 0) else { return }
        onVisiblePageChange?(pageNumber(for: currentPage, in: document))
    }

    private func pageNumber(for page: PDFPage, in document: PDFDocument) -> Int {
        for index in 0..<document.pageCount {
            if document.page(at: index) === page {
                return index + 1
            }
        }
        return 1
    }

    private func documentPointToTopOriginPagePoint(_ point: CGPoint, page: PDFPage, documentView: UIView) -> CGPoint? {
        let pointInPDFView = pdfView.convert(point, from: documentView)
        let pointOnPage = pdfView.convert(pointInPDFView, to: page)
        let pageBounds = page.bounds(for: .mediaBox)
        guard pointOnPage.x.isFinite, pointOnPage.y.isFinite else { return nil }
        return CGPoint(x: pointOnPage.x, y: pageBounds.maxY - pointOnPage.y)
    }

    private func labelEraseAreaFromTap(_ point: CGPoint, documentView: UIView) -> LabelEraseAreaState? {
        guard let document = pdfView.document,
              let page = pageContainingDocumentPoint(point, in: document, documentView: documentView),
              let topOriginPoint = documentPointToTopOriginPagePoint(point, page: page, documentView: documentView) else {
            return nil
        }
        let pageNumber = pageNumber(for: page, in: document)
        let bounds = page.bounds(for: .mediaBox)
        let width = 34.0
        let height = 18.0
        let left = max(0.0, min(Double(bounds.width) - width, Double(topOriginPoint.x) - (width / 2.0)))
        let top = max(0.0, min(Double(bounds.height) - height, Double(topOriginPoint.y) - (height / 2.0)))
        return LabelEraseAreaState(
            id: "label_erase_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(10))",
            page: pageNumber,
            rect: ManualRowRect(left: left, right: left + width, top: top, bottom: top + height)
        )
    }

    private func normalizedPageRect(
        pageBounds: CGRect,
        xLeft: CGFloat,
        xRight: CGFloat,
        yTop: CGFloat,
        yBottom: CGFloat,
        minimumSpan: CGFloat
    ) -> CGRect? {
        var left = max(0, min(pageBounds.width, min(xLeft, xRight)))
        var right = max(0, min(pageBounds.width, max(xLeft, xRight)))
        var topFromBackend = max(0, min(pageBounds.height, min(yTop, yBottom)))
        var bottomFromBackend = max(0, min(pageBounds.height, max(yTop, yBottom)))

        if right <= left || bottomFromBackend <= topFromBackend {
            return nil
        }

        if (right - left) < minimumSpan {
            right = min(pageBounds.width, left + minimumSpan)
        }
        if (bottomFromBackend - topFromBackend) < minimumSpan {
            bottomFromBackend = min(pageBounds.height, topFromBackend + minimumSpan)
        }
        if right <= left || bottomFromBackend <= topFromBackend {
            return nil
        }

        left = max(0, min(pageBounds.width, left))
        right = max(0, min(pageBounds.width, right))
        topFromBackend = max(0, min(pageBounds.height, topFromBackend))
        bottomFromBackend = max(0, min(pageBounds.height, bottomFromBackend))
        if right <= left || bottomFromBackend <= topFromBackend {
            return nil
        }

        // Backend rows are top-origin coordinates. PDFKit page space is bottom-origin.
        let rectHeight = bottomFromBackend - topFromBackend
        let pdfY = pageBounds.maxY - bottomFromBackend
        return CGRect(x: left, y: pdfY, width: right - left, height: rectHeight)
    }

    private func manualRowPageRect(_ row: ManualRowState, in page: PDFPage) -> CGRect? {
        normalizedPageRect(
            pageBounds: page.bounds(for: .mediaBox),
            xLeft: CGFloat(row.rect.left),
            xRight: CGFloat(row.rect.right),
            yTop: CGFloat(row.rect.top),
            yBottom: CGFloat(row.rect.bottom),
            minimumSpan: 8
        )
    }

    private func manualRowDocumentRect(_ row: ManualRowState, page: PDFPage, documentView: UIView) -> CGRect? {
        guard let pageRect = manualRowPageRect(row, in: page) else { return nil }
        let viewRect = pdfView.convert(pageRect, from: page)
        let docRect = documentView.convert(viewRect, from: pdfView)
        guard docRect.width > 0, docRect.height > 0 else { return nil }
        return docRect
    }

    private func manualRow(for rowID: String) -> ManualRowState? {
        currentManualEditor?.rows.first(where: { $0.manualRowId == rowID })
    }

    private func manualRowAtDocumentPoint(_ point: CGPoint, documentView: UIView) -> ManualRowState? {
        guard let manualEditor = currentManualEditor,
              let document = pdfView.document else { return nil }
        for row in manualEditor.rows.reversed() {
            guard row.page == manualEditor.activePage else { continue }
            let pageIndex = max(0, row.page - 1)
            guard let page = document.page(at: pageIndex),
                  let rect = manualRowDocumentRect(row, page: page, documentView: documentView) else { continue }
            if rect.insetBy(dx: -10, dy: -10).contains(point) {
                return row
            }
        }
        return nil
    }

    private func manualCutSelectionAtDocumentPoint(_ point: CGPoint, documentView: UIView) -> ManualSelectionState? {
        guard let manualEditor = currentManualEditor,
              let document = pdfView.document else { return nil }
        for row in manualEditor.rows.reversed() {
            guard row.page == manualEditor.activePage else { continue }
            let pageIndex = max(0, row.page - 1)
            guard let page = document.page(at: pageIndex),
                  let rect = manualRowDocumentRect(row, page: page, documentView: documentView),
                  rect.width > 0 else { continue }
            for (index, cutX) in row.cutXs.enumerated() {
                let ratio = (cutX - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
                let x = rect.minX + (rect.width * ratio)
                if abs(point.x - x) <= 14 && point.y >= rect.minY - 8 && point.y <= rect.maxY + 8 {
                    return ManualSelectionState(rowID: row.manualRowId, cutIndex: index)
                }
            }
        }
        return nil
    }

    private func autoRow(for rowID: String) -> AutoRowState? {
        currentAutoEditor?.rows.first(where: { $0.systemID == rowID })
    }

    private func autoRowPageRect(_ row: AutoRowState, in page: PDFPage) -> CGRect? {
        normalizedPageRect(
            pageBounds: page.bounds(for: .mediaBox),
            xLeft: CGFloat(row.rect.left),
            xRight: CGFloat(row.rect.right),
            yTop: CGFloat(row.rect.top),
            yBottom: CGFloat(row.rect.bottom),
            minimumSpan: 8
        )
    }

    private func autoRowDocumentRect(_ row: AutoRowState, page: PDFPage, documentView: UIView) -> CGRect? {
        guard let pageRect = autoRowPageRect(row, in: page) else { return nil }
        let viewRect = pdfView.convert(pageRect, from: page)
        let docRect = documentView.convert(viewRect, from: pdfView)
        guard docRect.width > 0, docRect.height > 0 else { return nil }
        return docRect
    }

    private func autoRowAtDocumentPoint(_ point: CGPoint, documentView: UIView) -> AutoRowState? {
        guard let autoEditor = currentAutoEditor,
              let document = pdfView.document else { return nil }
        for row in autoEditor.rows.reversed() {
            guard row.page == autoEditor.activePage else { continue }
            let pageIndex = max(0, row.page - 1)
            guard let page = document.page(at: pageIndex),
                  let rect = autoRowDocumentRect(row, page: page, documentView: documentView) else { continue }
            if rect.insetBy(dx: -10, dy: -10).contains(point) {
                return row
            }
        }
        return nil
    }

    private func autoSplitSelectionAtDocumentPoint(_ point: CGPoint, documentView: UIView) -> AutoSelectionState? {
        guard let autoEditor = currentAutoEditor,
              let document = pdfView.document else { return nil }
        for row in autoEditor.rows.reversed() {
            guard row.page == autoEditor.activePage else { continue }
            let pageIndex = max(0, row.page - 1)
            guard let page = document.page(at: pageIndex),
                  let rect = autoRowDocumentRect(row, page: page, documentView: documentView),
                  rect.width > 0,
                  row.boxes.count > 1 else { continue }
            for index in 0..<(row.boxes.count - 1) {
                let cutX = row.boxes[index].right
                let ratio = (cutX - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
                let x = rect.minX + (rect.width * ratio)
                if abs(point.x - x) <= 14 && point.y >= rect.minY - 8 && point.y <= rect.maxY + 8 {
                    return AutoSelectionState(rowID: row.systemID, splitIndex: index, measureID: nil)
                }
            }
        }
        return nil
    }

    private func autoBoxSelectionAtDocumentPoint(_ point: CGPoint, documentView: UIView) -> AutoSelectionState? {
        guard let autoEditor = currentAutoEditor,
              let document = pdfView.document else { return nil }
        for row in autoEditor.rows.reversed() {
            guard row.page == autoEditor.activePage else { continue }
            let pageIndex = max(0, row.page - 1)
            guard let page = document.page(at: pageIndex),
                  let rect = autoRowDocumentRect(row, page: page, documentView: documentView),
                  rect.width > 0 else { continue }
            for box in row.boxes {
                let leftRatio = (box.left - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
                let rightRatio = (box.right - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
                let boxRect = CGRect(
                    x: rect.minX + (rect.width * leftRatio),
                    y: rect.minY,
                    width: max(1, rect.width * (rightRatio - leftRatio)),
                    height: rect.height
                )
                if boxRect.contains(point) {
                    return AutoSelectionState(rowID: row.systemID, splitIndex: nil, measureID: box.measureID)
                }
            }
        }
        return nil
    }

    private func replaceAutoRow(_ row: AutoRowState) {
        guard let autoEditor = currentAutoEditor else { return }
        var rows = autoEditor.rows
        if let index = rows.firstIndex(where: { $0.systemID == row.systemID }) {
            rows[index] = row
        } else {
            rows.append(row)
        }
        rows.sort {
            if $0.page != $1.page { return $0.page < $1.page }
            if $0.rect.top != $1.rect.top { return $0.rect.top < $1.rect.top }
            if $0.rect.left != $1.rect.left { return $0.rect.left < $1.rect.left }
            return $0.systemID < $1.systemID
        }
        onAutoRowsChange?(rows)
    }

    private func manualHandleHit(
        at point: CGPoint,
        documentView: UIView
    ) -> (row: ManualRowState, corner: ManualCorner)? {
        guard let manualEditor = currentManualEditor,
              manualEditor.tool == .addRow || manualEditor.tool == .resizeRow,
              let selection = manualEditor.selection,
              selection.cutIndex == nil,
              let row = manualRow(for: selection.rowID),
              let document = pdfView.document else { return nil }
        let pageIndex = max(0, row.page - 1)
        guard let page = document.page(at: pageIndex),
              let rect = manualRowDocumentRect(row, page: page, documentView: documentView) else { return nil }
        let handles: [(ManualCorner, CGPoint)] = [
            (.topLeft, rect.origin),
            (.topRight, CGPoint(x: rect.maxX, y: rect.minY)),
            (.bottomLeft, CGPoint(x: rect.minX, y: rect.maxY)),
            (.bottomRight, CGPoint(x: rect.maxX, y: rect.maxY)),
        ]
        for (corner, handlePoint) in handles {
            if hypot(point.x - handlePoint.x, point.y - handlePoint.y) <= 18 {
                return (row, corner)
            }
        }
        return nil
    }

    private func autoHandleHit(
        at point: CGPoint,
        documentView: UIView
    ) -> (row: AutoRowState, corner: ManualCorner)? {
        guard let autoEditor = currentAutoEditor,
              autoEditor.tool == .resizeRow,
              let selection = autoEditor.selection,
              selection.splitIndex == nil,
              selection.measureID == nil,
              let row = autoRow(for: selection.rowID),
              let document = pdfView.document else { return nil }
        let pageIndex = max(0, row.page - 1)
        guard let page = document.page(at: pageIndex),
              let rect = autoRowDocumentRect(row, page: page, documentView: documentView) else { return nil }
        let handles: [(ManualCorner, CGPoint)] = [
            (.topLeft, rect.origin),
            (.topRight, CGPoint(x: rect.maxX, y: rect.minY)),
            (.bottomLeft, CGPoint(x: rect.minX, y: rect.maxY)),
            (.bottomRight, CGPoint(x: rect.maxX, y: rect.maxY)),
        ]
        for (corner, handlePoint) in handles {
            if hypot(point.x - handlePoint.x, point.y - handlePoint.y) <= 18 {
                return (row, corner)
            }
        }
        return nil
    }

    private func manualRowRectFromPoints(page: Int, start: CGPoint, end: CGPoint, staffKind: ManualStaffKind) -> ManualRowState? {
        let left = min(start.x, end.x)
        let right = max(start.x, end.x)
        let top = min(start.y, end.y)
        let bottom = max(start.y, end.y)
        guard right - left >= 12, bottom - top >= 10 else { return nil }
        return ManualRowState(
            manualRowId: "manual_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(10))",
            page: page,
            staffKind: staffKind,
            rect: ManualRowRect(left: left, right: right, top: top, bottom: bottom),
            cutXs: []
        )
    }

    private func replaceManualRow(_ row: ManualRowState) {
        guard let manualEditor = currentManualEditor else { return }
        var rows = manualEditor.rows
        if let index = rows.firstIndex(where: { $0.manualRowId == row.manualRowId }) {
            rows[index] = row
        } else {
            rows.append(row)
        }
        rows.sort(by: {
            if $0.page != $1.page { return $0.page < $1.page }
            if $0.rect.top != $1.rect.top { return $0.rect.top < $1.rect.top }
            if $0.rect.left != $1.rect.left { return $0.rect.left < $1.rect.left }
            return $0.manualRowId < $1.manualRowId
        })
        onManualRowsChange?(rows)
    }

    private func appendManualRow(_ row: ManualRowState) {
        guard let manualEditor = currentManualEditor else { return }
        var rows = manualEditor.rows
        rows.append(row)
        rows.sort(by: {
            if $0.page != $1.page { return $0.page < $1.page }
            if $0.rect.top != $1.rect.top { return $0.rect.top < $1.rect.top }
            if $0.rect.left != $1.rect.left { return $0.rect.left < $1.rect.left }
            return $0.manualRowId < $1.manualRowId
        })
        onManualRowsChange?(rows)
    }

    private func previewAutoRow(
        for rowID: String,
        startRect: ManualRowRect,
        currentPoint: CGPoint,
        page: PDFPage,
        documentView: UIView
    ) -> AutoRowState? {
        guard case let .resizeAuto(_, corner, _, _) = manualDragState,
              let row = autoRow(for: rowID),
              let point = documentPointToTopOriginPagePoint(currentPoint, page: page, documentView: documentView) else { return nil }
        let bounds = page.bounds(for: .mediaBox)
        let minSize = 10.0
        var left = startRect.left
        var right = startRect.right
        var top = startRect.top
        var bottom = startRect.bottom
        switch corner {
        case .topLeft:
            left = min(max(0, point.x), right - minSize)
            top = min(max(0, point.y), bottom - minSize)
        case .topRight:
            right = max(min(bounds.width, point.x), left + minSize)
            top = min(max(0, point.y), bottom - minSize)
        case .bottomLeft:
            left = min(max(0, point.x), right - minSize)
            bottom = max(min(bounds.height, point.y), top + minSize)
        case .bottomRight:
            right = max(min(bounds.width, point.x), left + minSize)
            bottom = max(min(bounds.height, point.y), top + minSize)
        }

        let keptSplits = row.boxes.dropLast().map(\.right).filter { $0 > left + 2 && $0 < right - 2 }.sorted()
        let boundaries = [left] + keptSplits + [right]
        guard boundaries.count >= 2 else { return nil }

        var rebuiltBoxes: [AutoBoxState] = []
        for index in 0..<(boundaries.count - 1) {
            let boxLeft = boundaries[index]
            let boxRight = boundaries[index + 1]
            guard boxRight - boxLeft >= 2 else { continue }
            let merged = row.boxes.filter { $0.right > boxLeft && $0.left < boxRight }
            guard let first = merged.first else { continue }
            rebuiltBoxes.append(
                AutoBoxState(
                    measureID: first.measureID,
                    left: boxLeft,
                    right: boxRight,
                    excludedFromCounting: merged.contains(where: { $0.excludedFromCounting })
                )
            )
        }
        guard !rebuiltBoxes.isEmpty else { return nil }

        return AutoRowState(
            systemID: row.systemID,
            page: row.page,
            rect: ManualRowRect(left: left, right: right, top: top, bottom: bottom),
            boxes: rebuiltBoxes
        )
    }

    private func addSplitDot(
        at center: CGPoint,
        color: UIColor = .systemBlue,
        diameter: CGFloat = 4
    ) {
        let dotLayer = CAShapeLayer()
        let dotRect = CGRect(
            x: center.x - (diameter / 2),
            y: center.y - (diameter / 2),
            width: diameter,
            height: diameter
        )
        dotLayer.frame = dotRect
        dotLayer.path = UIBezierPath(ovalIn: dotLayer.bounds).cgPath
        dotLayer.fillColor = color.cgColor
        dotLayer.strokeColor = UIColor.white.cgColor
        dotLayer.lineWidth = 0.8
        manualLayer.addSublayer(dotLayer)
    }

    private func addManualCutLine(
        from start: CGPoint,
        to end: CGPoint,
        color: UIColor,
        foregroundWidth: CGFloat,
        dotDiameter: CGFloat
    ) {
        let foregroundLayer = CAShapeLayer()
        let foregroundPath = UIBezierPath()
        foregroundPath.move(to: start)
        foregroundPath.addLine(to: end)
        foregroundLayer.path = foregroundPath.cgPath
        foregroundLayer.strokeColor = color.cgColor
        foregroundLayer.lineWidth = foregroundWidth
        manualLayer.addSublayer(foregroundLayer)
        addSplitDot(at: CGPoint(x: end.x, y: end.y + 8), color: .systemBlue, diameter: dotDiameter)
    }

    private func drawManualRows(in documentView: UIView) {
        guard let manualEditor = currentManualEditor,
              let document = pdfView.document else { return }
        for row in manualEditor.rows where row.page == manualEditor.activePage {
            let pageIndex = max(0, row.page - 1)
            guard let page = document.page(at: pageIndex),
                  let rect = manualRowDocumentRect(row, page: page, documentView: documentView) else { continue }

            let selected = manualEditor.selection?.rowID == row.manualRowId
            let rowLayer = CAShapeLayer()
            rowLayer.frame = rect
            rowLayer.path = UIBezierPath(roundedRect: rowLayer.bounds, cornerRadius: 4).cgPath
            rowLayer.fillColor = UIColor.clear.cgColor
            rowLayer.strokeColor = (selected ? UIColor.systemBlue : UIColor.systemGreen.withAlphaComponent(0.95)).cgColor
            rowLayer.lineWidth = selected ? 2.2 : 1.6
            manualLayer.addSublayer(rowLayer)

            for (index, cutX) in row.cutXs.enumerated() {
                let ratio = (cutX - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
                let x = rect.minX + (rect.width * ratio)
                let isSelectedCut = manualEditor.selection?.rowID == row.manualRowId && manualEditor.selection?.cutIndex == index
                addManualCutLine(
                    from: CGPoint(x: x, y: rect.minY),
                    to: CGPoint(x: x, y: rect.maxY),
                    color: isSelectedCut ? .systemBlue : .systemGreen,
                    foregroundWidth: isSelectedCut ? 3.0 : 2.0,
                    dotDiameter: isSelectedCut ? 5 : 4
                )
            }

            if selected,
               manualEditor.selection?.cutIndex == nil,
               (manualEditor.tool == .addRow || manualEditor.tool == .resizeRow) {
                let handlePoints = [
                    CGPoint(x: rect.minX, y: rect.minY),
                    CGPoint(x: rect.maxX, y: rect.minY),
                    CGPoint(x: rect.minX, y: rect.maxY),
                    CGPoint(x: rect.maxX, y: rect.maxY),
                ]
                for point in handlePoints {
                    let handleLayer = CAShapeLayer()
                    let handleRect = CGRect(x: point.x - 8, y: point.y - 8, width: 16, height: 16)
                    handleLayer.frame = handleRect
                    handleLayer.path = UIBezierPath(ovalIn: handleLayer.bounds).cgPath
                    handleLayer.fillColor = UIColor.systemBlue.cgColor
                    handleLayer.strokeColor = UIColor.white.cgColor
                    handleLayer.lineWidth = 1.2
                    manualLayer.addSublayer(handleLayer)
                }
            }
        }
    }

    private func labelErasePageRect(_ area: LabelEraseAreaState, in page: PDFPage) -> CGRect? {
        normalizedPageRect(
            pageBounds: page.bounds(for: .mediaBox),
            xLeft: CGFloat(area.rect.left),
            xRight: CGFloat(area.rect.right),
            yTop: CGFloat(area.rect.top),
            yBottom: CGFloat(area.rect.bottom),
            minimumSpan: 4
        )
    }

    private func labelEraseDocumentRect(_ area: LabelEraseAreaState, page: PDFPage, documentView: UIView) -> CGRect? {
        guard let pageRect = labelErasePageRect(area, in: page) else { return nil }
        let viewRect = pdfView.convert(pageRect, from: page)
        let docRect = documentView.convert(viewRect, from: pdfView)
        guard docRect.width > 0, docRect.height > 0 else { return nil }
        return docRect
    }

    private func drawPendingLabelEraseArea(in documentView: UIView) {
        guard let area = currentPendingLabelEraseArea,
              let document = pdfView.document,
              let page = document.page(at: max(0, area.page - 1)),
              let rect = labelEraseDocumentRect(area, page: page, documentView: documentView) else { return }
        let layer = CAShapeLayer()
        layer.frame = rect
        layer.path = UIBezierPath(roundedRect: layer.bounds, cornerRadius: 2).cgPath
        layer.fillColor = UIColor.systemBlue.withAlphaComponent(0.16).cgColor
        layer.strokeColor = UIColor.systemBlue.withAlphaComponent(0.95).cgColor
        layer.lineWidth = 2.0
        manualLayer.addSublayer(layer)
    }

    private func drawManualPreview(in documentView: UIView) {
        guard let drag = manualDragState,
              let manualEditor = currentManualEditor,
              let document = pdfView.document else { return }
        switch drag {
        case .addRow(let page, let start, let current):
            guard let preview = manualRowRectFromPoints(page: page, start: start, end: current, staffKind: manualEditor.defaultStaffKind) else { return }
            let pageIndex = max(0, page - 1)
            guard let pdfPage = document.page(at: pageIndex),
                  let rect = manualRowDocumentRect(preview, page: pdfPage, documentView: documentView) else { return }
            let previewLayer = CAShapeLayer()
            previewLayer.frame = rect
            previewLayer.path = UIBezierPath(roundedRect: previewLayer.bounds, cornerRadius: 4).cgPath
            previewLayer.fillColor = UIColor.clear.cgColor
            previewLayer.strokeColor = UIColor.systemBlue.withAlphaComponent(0.95).cgColor
            previewLayer.lineWidth = 1.8
            manualLayer.addSublayer(previewLayer)

        case .addCut(let rowID, let currentX):
            guard let row = manualRow(for: rowID) else { return }
            let pageIndex = max(0, row.page - 1)
            guard let pdfPage = document.page(at: pageIndex),
                  let rect = manualRowDocumentRect(row, page: pdfPage, documentView: documentView) else { return }
            let ratio = (currentX - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
            let x = rect.minX + (rect.width * ratio)
            addManualCutLine(
                from: CGPoint(x: x, y: rect.minY),
                to: CGPoint(x: x, y: rect.maxY),
                color: .systemBlue,
                foregroundWidth: 2.5,
                dotDiameter: 4
            )

        case .resize(let rowID, _, let startRect, let currentPoint):
            guard let selectedRow = manualRow(for: rowID),
                  let page = document.page(at: max(0, selectedRow.page - 1)),
                  let previewRect = previewRowRect(for: rowID, startRect: startRect, currentPoint: currentPoint, page: page, documentView: documentView),
                  let rect = manualRowDocumentRect(previewRect, page: page, documentView: documentView) else { return }
            let previewLayer = CAShapeLayer()
            previewLayer.frame = rect
            previewLayer.path = UIBezierPath(roundedRect: previewLayer.bounds, cornerRadius: 4).cgPath
            previewLayer.fillColor = UIColor.clear.cgColor
            previewLayer.strokeColor = UIColor.systemBlue.cgColor
            previewLayer.lineWidth = 2.2
            manualLayer.addSublayer(previewLayer)
        case .addAutoCut, .resizeAuto:
            break
        }
    }

    private func drawAutoRows(in documentView: UIView) {
        guard let autoEditor = currentAutoEditor,
              let document = pdfView.document else { return }
        for row in autoEditor.rows where row.page == autoEditor.activePage {
            let pageIndex = max(0, row.page - 1)
            guard let page = document.page(at: pageIndex),
                  let rect = autoRowDocumentRect(row, page: page, documentView: documentView) else { continue }

            let rowSelected =
                autoEditor.selection?.rowID == row.systemID
                && autoEditor.selection?.splitIndex == nil
                && autoEditor.selection?.measureID == nil
            let rowLayer = CAShapeLayer()
            rowLayer.frame = rect
            rowLayer.path = UIBezierPath(roundedRect: rowLayer.bounds, cornerRadius: 4).cgPath
            rowLayer.fillColor = UIColor.clear.cgColor
            rowLayer.strokeColor = (rowSelected ? UIColor.systemBlue : UIColor.systemGreen.withAlphaComponent(0.95)).cgColor
            rowLayer.lineWidth = rowSelected ? 2.2 : 1.6
            manualLayer.addSublayer(rowLayer)

            for box in row.boxes {
                let leftRatio = (box.left - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
                let rightRatio = (box.right - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
                let boxRect = CGRect(
                    x: rect.minX + (rect.width * leftRatio),
                    y: rect.minY,
                    width: max(1, rect.width * (rightRatio - leftRatio)),
                    height: rect.height
                )
                let isSelectedBox = autoEditor.selection?.rowID == row.systemID && autoEditor.selection?.measureID == box.measureID
                if box.excludedFromCounting || isSelectedBox {
                    let boxLayer = CAShapeLayer()
                    boxLayer.frame = boxRect
                    boxLayer.path = UIBezierPath(roundedRect: boxLayer.bounds, cornerRadius: 4).cgPath
                    if box.excludedFromCounting {
                        boxLayer.fillColor = EditColorPalette.excluded.fill.withAlphaComponent(isSelectedBox ? 0.32 : 0.22).cgColor
                    } else {
                        boxLayer.fillColor = UIColor.clear.cgColor
                    }
                    boxLayer.strokeColor = (
                        isSelectedBox
                        ? UIColor.systemBlue
                        : (box.excludedFromCounting ? EditColorPalette.excluded.stroke : UIColor.systemGreen.withAlphaComponent(0.9))
                    ).cgColor
                    boxLayer.lineWidth = isSelectedBox ? 2.6 : 1.6
                    manualLayer.addSublayer(boxLayer)
                }
            }

            if row.boxes.count > 1 {
                for index in 0..<(row.boxes.count - 1) {
                    let cutX = row.boxes[index].right
                    let ratio = (cutX - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
                    let x = rect.minX + (rect.width * ratio)
                    let isSelectedCut = autoEditor.selection?.rowID == row.systemID && autoEditor.selection?.splitIndex == index
                    addManualCutLine(
                        from: CGPoint(x: x, y: rect.minY),
                        to: CGPoint(x: x, y: rect.maxY),
                        color: isSelectedCut ? .systemBlue : .systemGreen,
                        foregroundWidth: isSelectedCut ? 3.0 : 2.0,
                        dotDiameter: isSelectedCut ? 5 : 4
                    )
                }
            }

            if rowSelected,
               autoEditor.selection?.splitIndex == nil,
               autoEditor.selection?.measureID == nil,
               autoEditor.tool == .resizeRow {
                let handlePoints = [
                    CGPoint(x: rect.minX, y: rect.minY),
                    CGPoint(x: rect.maxX, y: rect.minY),
                    CGPoint(x: rect.minX, y: rect.maxY),
                    CGPoint(x: rect.maxX, y: rect.maxY),
                ]
                for point in handlePoints {
                    let handleLayer = CAShapeLayer()
                    let handleRect = CGRect(x: point.x - 8, y: point.y - 8, width: 16, height: 16)
                    handleLayer.frame = handleRect
                    handleLayer.path = UIBezierPath(ovalIn: handleLayer.bounds).cgPath
                    handleLayer.fillColor = UIColor.systemBlue.cgColor
                    handleLayer.strokeColor = UIColor.white.cgColor
                    handleLayer.lineWidth = 1.2
                    manualLayer.addSublayer(handleLayer)
                }
            }
        }
    }

    private func drawAutoPreview(in documentView: UIView) {
        guard let drag = manualDragState,
              let autoEditor = currentAutoEditor,
              let document = pdfView.document else { return }
        switch drag {
        case .addAutoCut(let rowID, _, let currentX):
            guard let row = autoRow(for: rowID) else { return }
            let pageIndex = max(0, row.page - 1)
            guard let pdfPage = document.page(at: pageIndex),
                  let rect = autoRowDocumentRect(row, page: pdfPage, documentView: documentView) else { return }
            let ratio = (currentX - row.rect.left) / max(1.0, row.rect.right - row.rect.left)
            let x = rect.minX + (rect.width * ratio)
            addManualCutLine(
                from: CGPoint(x: x, y: rect.minY),
                to: CGPoint(x: x, y: rect.maxY),
                color: .systemBlue,
                foregroundWidth: 2.5,
                dotDiameter: 4
            )
        case .resizeAuto(let rowID, _, let startRect, let currentPoint):
            guard let selectedRow = autoRow(for: rowID),
                  let page = document.page(at: max(0, selectedRow.page - 1)),
                  let previewRow = previewAutoRow(for: rowID, startRect: startRect, currentPoint: currentPoint, page: page, documentView: documentView),
                  let rect = autoRowDocumentRect(previewRow, page: page, documentView: documentView) else { return }
            let previewLayer = CAShapeLayer()
            previewLayer.frame = rect
            previewLayer.path = UIBezierPath(roundedRect: previewLayer.bounds, cornerRadius: 4).cgPath
            previewLayer.fillColor = UIColor.clear.cgColor
            previewLayer.strokeColor = UIColor.systemBlue.cgColor
            previewLayer.lineWidth = 2.2
            manualLayer.addSublayer(previewLayer)
        default:
            _ = autoEditor
        }
    }

    private func previewRowRect(
        for rowID: String,
        startRect: ManualRowRect,
        currentPoint: CGPoint,
        page: PDFPage,
        documentView: UIView
    ) -> ManualRowState? {
        guard case let .resize(_, corner, _, _) = manualDragState,
              let row = manualRow(for: rowID),
              let point = documentPointToTopOriginPagePoint(currentPoint, page: page, documentView: documentView) else { return nil }
        let bounds = page.bounds(for: .mediaBox)
        let minSize = 10.0
        var left = startRect.left
        var right = startRect.right
        var top = startRect.top
        var bottom = startRect.bottom
        switch corner {
        case .topLeft:
            left = min(max(0, point.x), right - minSize)
            top = min(max(0, point.y), bottom - minSize)
        case .topRight:
            right = max(min(bounds.width, point.x), left + minSize)
            top = min(max(0, point.y), bottom - minSize)
        case .bottomLeft:
            left = min(max(0, point.x), right - minSize)
            bottom = max(min(bounds.height, point.y), top + minSize)
        case .bottomRight:
            right = max(min(bounds.width, point.x), left + minSize)
            bottom = max(min(bounds.height, point.y), top + minSize)
        }
        return ManualRowState(
            manualRowId: row.manualRowId,
            page: row.page,
            staffKind: row.staffKind,
            rect: ManualRowRect(left: left, right: right, top: top, bottom: bottom),
            cutXs: row.cutXs.filter { $0 > left && $0 < right }.sorted()
        )
    }

    private func buildPrimaryRows(in doc: PDFDocument) -> [OverlayRow] {
        guard !currentMeasures.isEmpty else { return [] }

        var rows: [OverlayRow] = []
        for measure in currentMeasures {
            let systemID = measure.system_id.trimmingCharacters(in: .whitespacesAndNewlines)
            let measureID = measure.measure_id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? measure.id
            if systemID.isEmpty || measureID.isEmpty { continue }
            let page = max(1, measure.page)
            let pageIndex = max(0, page - 1)
            guard let pdfPage = doc.page(at: pageIndex) else { continue }
            let pageBounds = pdfPage.bounds(for: .mediaBox)
            guard let pageRect = normalizedPageRect(
                pageBounds: pageBounds,
                xLeft: CGFloat(measure.x_left),
                xRight: CGFloat(measure.x_right),
                yTop: CGFloat(measure.y_top),
                yBottom: CGFloat(measure.y_bottom),
                minimumSpan: 8
            ) else { continue }

            let local = measure.measure_local_index ?? -1
            rows.append(
                OverlayRow(
                    rowID: "measure|\(page)|\(systemID)|\(local)|\(Int(pageRect.minX.rounded()))",
                    measureID: measureID,
                    systemID: systemID,
                    page: page,
                    pageRect: pageRect
                )
            )
        }
        return rows.sorted(by: { lhs, rhs in
            if lhs.page != rhs.page { return lhs.page < rhs.page }
            if lhs.pageRect.minY != rhs.pageRect.minY { return lhs.pageRect.minY < rhs.pageRect.minY }
            if lhs.pageRect.minX != rhs.pageRect.minX { return lhs.pageRect.minX < rhs.pageRect.minX }
            return lhs.rowID < rhs.rowID
        })
    }

    private func logRowsIfNeeded(_ rows: [OverlayRow], token: UUID) {
        guard FrontendDebugConfig.overlayLoggingEnabled else { return }
        let signature = "\(token.uuidString)|\(currentLabelsMode.rawValue)|\(rows.count)"
        if signature == lastOverlayLogSignature { return }
        lastOverlayLogSignature = signature

        var pageRows: [Int: [OverlayRow]] = [:]
        for row in rows {
            pageRows[row.page, default: []].append(row)
        }
        let pages = pageRows.keys.sorted()
        for page in pages {
            guard let rowsOnPage = pageRows[page] else { continue }
            let firstRows = rowsOnPage.prefix(FrontendDebugConfig.overlayLogMaxRowsPerPage)
            for row in firstRows {
                let r = row.pageRect
                print(
                    "OVERLAY_RECT token=\(token.uuidString.prefix(8)) mode=\(currentLabelsMode.rawValue) " +
                    "kind=measure page=\(row.page) system_id=\(row.systemID) " +
                    "x=\(String(format: "%.2f", r.minX)) y=\(String(format: "%.2f", r.minY)) " +
                    "w=\(String(format: "%.2f", r.width)) h=\(String(format: "%.2f", r.height))"
                )
            }
        }
    }

    private func activeEditStyles(for measureID: String) -> [EditColorStyle] {
        var styles: [EditColorStyle] = []
        if currentMeasures.first(where: { $0.id == measureID })?.excluded_from_counting == true {
            styles.append(EditColorPalette.excluded)
        }
        styles.append(contentsOf: SavedMeasureEditKind.displayOrder.compactMap { kind -> EditColorStyle? in
            switch kind {
            case .measureNumber:
                return currentMeasureNumberOverrideIDs.contains(measureID) ? EditColorPalette.measureNumber : nil
            case .pickup:
                return currentPickupAnchorIDs.contains(measureID) ? EditColorPalette.pickup : nil
            case .rest:
                return currentRestAnchorIDs.contains(measureID) ? EditColorPalette.rest : nil
            }
        })
        if currentEnding1AnchorIDs.contains(measureID) {
            styles.append(EditColorPalette.ending1)
        }
        if currentEnding2AnchorIDs.contains(measureID) {
            styles.append(EditColorPalette.ending2)
        }
        if currentPendingEnding1IDs.contains(measureID) {
            styles.append(EditColorPalette.ending1)
        }
        if currentPendingEnding2IDs.contains(measureID) {
            styles.append(EditColorPalette.ending2)
        }
        if styles.isEmpty && currentAISuggestionMeasureIDs.contains(measureID) {
            styles.append(EditColorPalette.aiSuggestion)
        }
        if styles.isEmpty {
            styles.append(EditColorPalette.normal)
        }
        return styles
    }

    private func addFillLayers(to container: CAShapeLayer, styles: [EditColorStyle], cornerRadius: CGFloat) {
        let contentLayer = CALayer()
        contentLayer.frame = container.bounds

        let maskLayer = CAShapeLayer()
        maskLayer.frame = contentLayer.bounds
        maskLayer.path = UIBezierPath(roundedRect: contentLayer.bounds, cornerRadius: cornerRadius).cgPath
        contentLayer.mask = maskLayer

        if styles.count <= 1 {
            let fillLayer = CALayer()
            fillLayer.frame = contentLayer.bounds
            fillLayer.backgroundColor = styles[0].fill.cgColor
            contentLayer.addSublayer(fillLayer)
        } else {
            let stripeHeight = contentLayer.bounds.height / CGFloat(styles.count)
            for (index, style) in styles.enumerated() {
                let y = CGFloat(index) * stripeHeight
                let height = index == styles.count - 1 ? contentLayer.bounds.height - y : stripeHeight
                let stripeLayer = CALayer()
                stripeLayer.frame = CGRect(x: 0, y: y, width: contentLayer.bounds.width, height: height)
                stripeLayer.backgroundColor = style.fill.cgColor
                contentLayer.addSublayer(stripeLayer)
            }
        }

        container.addSublayer(contentLayer)
    }

    private func addBorderLayer(to container: CAShapeLayer, styles: [EditColorStyle], cornerRadius: CGFloat) {
        let borderLayer = CAShapeLayer()
        borderLayer.frame = container.bounds
        borderLayer.path = UIBezierPath(roundedRect: borderLayer.bounds, cornerRadius: cornerRadius).cgPath
        borderLayer.fillColor = UIColor.clear.cgColor
        if styles.count == 1 {
            borderLayer.strokeColor = styles[0].stroke.cgColor
        } else {
            borderLayer.strokeColor = UIColor.secondaryLabel.withAlphaComponent(0.28).cgColor
        }
        borderLayer.lineWidth = 1.1
        container.addSublayer(borderLayer)
    }

    private func drawRows(_ rows: [OverlayRow], in doc: PDFDocument, documentView: UIView) -> Int {
        var drawn = 0
        var rowsByPageAndSystem: [String: [CGRect]] = [:]
        for row in rows {
            let pageIndex = max(0, row.page - 1)
            guard let page = doc.page(at: pageIndex) else { continue }
            let viewRect = pdfView.convert(row.pageRect, from: page)
            let docRect = documentView.convert(viewRect, from: pdfView)
            if docRect.width <= 0 || docRect.height <= 0 { continue }
            rowsByPageAndSystem["\(row.page)|\(row.systemID)", default: []].append(docRect)

            let layer = CAShapeLayer()
            layer.frame = docRect
            let cornerRadius: CGFloat = 3
            layer.path = UIBezierPath(roundedRect: layer.bounds, cornerRadius: cornerRadius).cgPath
            layer.fillColor = UIColor.clear.cgColor
            layer.strokeColor = UIColor.clear.cgColor
            layer.name = row.rowID
            layer.setValue(row.measureID, forKey: "measureID")
            layer.setValue(row.systemID, forKey: "systemID")

            let styles = activeEditStyles(for: row.measureID)
            addFillLayers(to: layer, styles: styles, cornerRadius: cornerRadius)
            addBorderLayer(to: layer, styles: styles, cornerRadius: cornerRadius)

            overlayLayer.addSublayer(layer)
            drawn += 1
        }
        for rects in rowsByPageAndSystem.values {
            let sortedRects = rects.sorted {
                if $0.minY != $1.minY { return $0.minY < $1.minY }
                return $0.minX < $1.minX
            }
            guard sortedRects.count > 1 else { continue }
            for rect in sortedRects.dropFirst() {
                let dotLayer = CAShapeLayer()
                let diameter: CGFloat = 4
                let center = CGPoint(x: rect.minX, y: rect.maxY + 8)
                let dotRect = CGRect(
                    x: center.x - (diameter / 2),
                    y: center.y - (diameter / 2),
                    width: diameter,
                    height: diameter
                )
                dotLayer.frame = dotRect
                dotLayer.path = UIBezierPath(ovalIn: dotLayer.bounds).cgPath
                dotLayer.fillColor = UIColor.systemBlue.cgColor
                dotLayer.strokeColor = UIColor.white.cgColor
                dotLayer.lineWidth = 0.8
                overlayLayer.addSublayer(dotLayer)
            }
        }
        return drawn
    }

    @objc private func redrawOverlays() {
        guard let documentView = pdfView.documentView else { return }

        if overlayLayer.superlayer !== documentView.layer {
            overlayLayer.removeFromSuperlayer()
            documentView.layer.addSublayer(overlayLayer)
        }
        if manualLayer.superlayer !== documentView.layer {
            manualLayer.removeFromSuperlayer()
            documentView.layer.addSublayer(manualLayer)
        }

        overlayLayer.frame = documentView.bounds
        overlayLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
        manualLayer.frame = documentView.bounds
        manualLayer.sublayers?.forEach { $0.removeFromSuperlayer() }

        guard let doc = pdfView.document else { return }
        let token = currentSnapshotToken

        let rows = buildPrimaryRows(in: doc)
        logRowsIfNeeded(rows, token: token)
        guard token == currentSnapshotToken else { return }
        let drawn = drawRows(rows, in: doc, documentView: documentView)
        drawAutoRows(in: documentView)
        drawAutoPreview(in: documentView)
        drawManualRows(in: documentView)
        drawPendingLabelEraseArea(in: documentView)
        drawManualPreview(in: documentView)
        notifyVisiblePageIfNeeded()
        onOverlayCount?(drawn)
    }

    @objc private func handleTap(_ sender: UITapGestureRecognizer) {
        guard sender.state == .ended else { return }
        guard let documentView = pdfView.documentView else { return }

        let tap = sender.location(in: documentView)
        if let tool = currentManualEditor?.tool ?? currentAutoEditor?.tool {
            switch tool {
            case .addRow:
                if let row = manualRowAtDocumentPoint(tap, documentView: documentView) {
                    onManualSelectionChange?(ManualSelectionState(rowID: row.manualRowId, cutIndex: nil))
                    onAutoSelectionChange?(nil)
                    return
                }
            case .addMeasures:
                if let row = manualRowAtDocumentPoint(tap, documentView: documentView) {
                    onManualSelectionChange?(ManualSelectionState(rowID: row.manualRowId, cutIndex: nil))
                    onAutoSelectionChange?(nil)
                    return
                }
                if let row = autoRowAtDocumentPoint(tap, documentView: documentView) {
                    onAutoSelectionChange?(AutoSelectionState(rowID: row.systemID, splitIndex: nil, measureID: nil))
                    onManualSelectionChange?(nil)
                    return
                }
            case .resizeRow:
                if let row = manualRowAtDocumentPoint(tap, documentView: documentView) {
                    onManualSelectionChange?(ManualSelectionState(rowID: row.manualRowId, cutIndex: nil))
                    onAutoSelectionChange?(nil)
                    return
                }
                if let row = autoRowAtDocumentPoint(tap, documentView: documentView) {
                    onAutoSelectionChange?(AutoSelectionState(rowID: row.systemID, splitIndex: nil, measureID: nil))
                    onManualSelectionChange?(nil)
                    return
                }
            case .delete:
                if let cutSelection = manualCutSelectionAtDocumentPoint(tap, documentView: documentView) {
                    onManualSelectionChange?(cutSelection)
                    onAutoSelectionChange?(nil)
                    return
                }
                if let row = manualRowAtDocumentPoint(tap, documentView: documentView) {
                    onManualSelectionChange?(ManualSelectionState(rowID: row.manualRowId, cutIndex: nil))
                    onAutoSelectionChange?(nil)
                    return
                }
                if let splitSelection = autoSplitSelectionAtDocumentPoint(tap, documentView: documentView) {
                    onAutoSelectionChange?(splitSelection)
                    onManualSelectionChange?(nil)
                    return
                }
                if let boxSelection = autoBoxSelectionAtDocumentPoint(tap, documentView: documentView) {
                    onAutoSelectionChange?(boxSelection)
                    onManualSelectionChange?(nil)
                    return
                }
            case .exclude:
                if let row = autoRowAtDocumentPoint(tap, documentView: documentView) {
                    if let currentSelection = currentAutoEditor?.selection,
                       currentSelection.rowID == row.systemID,
                       currentSelection.splitIndex == nil,
                       let boxSelection = autoBoxSelectionAtDocumentPoint(tap, documentView: documentView),
                       boxSelection.rowID == row.systemID {
                        onAutoSelectionChange?(boxSelection)
                    } else {
                        onAutoSelectionChange?(AutoSelectionState(rowID: row.systemID, splitIndex: nil, measureID: nil))
                    }
                    onManualSelectionChange?(nil)
                    return
                }
            case .removeLabel:
                guard let area = labelEraseAreaFromTap(tap, documentView: documentView),
                      area.page == currentManualEditor?.activePage else {
                    onLabelEraseAreaChange?(nil)
                    return
                }
                onLabelEraseAreaChange?(area)
                onManualSelectionChange?(nil)
                onAutoSelectionChange?(nil)
                return
            }
            onManualSelectionChange?(nil)
            onAutoSelectionChange?(nil)
            return
        }
        for sub in overlayLayer.sublayers ?? [] {
            guard let shape = sub as? CAShapeLayer,
                  shape.frame.contains(tap),
                  let id = (shape.value(forKey: "measureID") as? String) ?? shape.name,
                  let measure = currentMeasures.first(where: { $0.id == id }) else {
                continue
            }
            onSelectMeasure?(measure)
            return
        }
    }

    @objc private func handlePan(_ sender: UIPanGestureRecognizer) {
        guard let documentView = pdfView.documentView,
              let document = pdfView.document else { return }
        let location = sender.location(in: documentView)
        switch sender.state {
        case .began:
            if let handle = manualHandleHit(at: location, documentView: documentView) {
                setManualInteractionLocked(true)
                manualDragState = .resize(
                    rowID: handle.row.manualRowId,
                    corner: handle.corner,
                    startRect: handle.row.rect,
                    currentPoint: location
                )
                redrawOverlays()
                return
            }
            if let handle = autoHandleHit(at: location, documentView: documentView) {
                setManualInteractionLocked(true)
                manualDragState = .resizeAuto(
                    rowID: handle.row.systemID,
                    corner: handle.corner,
                    startRect: handle.row.rect,
                    currentPoint: location
                )
                redrawOverlays()
                return
            }

            guard let tool = currentManualEditor?.tool ?? currentAutoEditor?.tool else { return }
            switch tool {
            case .addRow:
                guard let manualEditor = currentManualEditor else { return }
                if let row = manualRowAtDocumentPoint(location, documentView: documentView) {
                    onManualSelectionChange?(ManualSelectionState(rowID: row.manualRowId, cutIndex: nil))
                    onAutoSelectionChange?(nil)
                    redrawOverlays()
                    return
                }
                if autoRowAtDocumentPoint(location, documentView: documentView) != nil {
                    return
                }
                guard let page = pageContainingDocumentPoint(location, in: document, documentView: documentView) else { return }
                let pageNumber = pageNumber(for: page, in: document)
                guard pageNumber == manualEditor.activePage,
                      let start = documentPointToTopOriginPagePoint(location, page: page, documentView: documentView) else { return }
                setManualInteractionLocked(true)
                manualDragState = .addRow(page: pageNumber, start: start, current: start)
            case .addMeasures:
                if let manualEditor = currentManualEditor,
                   let row = manualRowAtDocumentPoint(location, documentView: documentView),
                   let page = document.page(at: max(0, row.page - 1)),
                   row.page == manualEditor.activePage,
                   let point = documentPointToTopOriginPagePoint(location, page: page, documentView: documentView),
                   point.x >= row.rect.left,
                   point.x <= row.rect.right,
                   point.y >= row.rect.top,
                   point.y <= row.rect.bottom {
                    if manualEditor.selection?.rowID != row.manualRowId || manualEditor.selection?.cutIndex != nil {
                        onManualSelectionChange?(ManualSelectionState(rowID: row.manualRowId, cutIndex: nil))
                        onAutoSelectionChange?(nil)
                    }
                    let clampedX = min(max(point.x, row.rect.left + 1), row.rect.right - 1)
                    setManualInteractionLocked(true)
                    manualDragState = .addCut(rowID: row.manualRowId, currentX: clampedX)
                    redrawOverlays()
                    return
                }
                guard let autoEditor = currentAutoEditor,
                      let row = autoRowAtDocumentPoint(location, documentView: documentView),
                      let page = document.page(at: max(0, row.page - 1)),
                      row.page == autoEditor.activePage,
                      let point = documentPointToTopOriginPagePoint(location, page: page, documentView: documentView),
                      let boxIndex = row.boxes.firstIndex(where: { point.x > $0.left && point.x < $0.right }) else { return }
                let box = row.boxes[boxIndex]
                let clampedX = min(max(point.x, box.left + 1), box.right - 1)
                onAutoSelectionChange?(AutoSelectionState(rowID: row.systemID, splitIndex: nil, measureID: nil))
                onManualSelectionChange?(nil)
                setManualInteractionLocked(true)
                manualDragState = .addAutoCut(rowID: row.systemID, boxIndex: boxIndex, currentX: clampedX)
            case .resizeRow, .delete, .exclude, .removeLabel:
                return
            }
            redrawOverlays()

        case .changed:
            guard let drag = manualDragState else { return }
            switch drag {
            case .addRow(let page, let start, _):
                guard let pdfPage = document.page(at: max(0, page - 1)),
                      let current = documentPointToTopOriginPagePoint(location, page: pdfPage, documentView: documentView) else { return }
                manualDragState = .addRow(page: page, start: start, current: current)
            case .addCut(let rowID, _):
                guard let row = manualRow(for: rowID),
                      let pdfPage = document.page(at: max(0, row.page - 1)),
                      let current = documentPointToTopOriginPagePoint(location, page: pdfPage, documentView: documentView) else { return }
                let clampedX = min(max(current.x, row.rect.left + 1), row.rect.right - 1)
                manualDragState = .addCut(rowID: rowID, currentX: clampedX)
            case .resize(let rowID, let corner, let startRect, _):
                manualDragState = .resize(rowID: rowID, corner: corner, startRect: startRect, currentPoint: location)
            case .addAutoCut(let rowID, _, _):
                guard let row = autoRow(for: rowID),
                      let pdfPage = document.page(at: max(0, row.page - 1)),
                      let current = documentPointToTopOriginPagePoint(location, page: pdfPage, documentView: documentView),
                      let boxIndex = row.boxes.firstIndex(where: { current.x > $0.left && current.x < $0.right }) ?? row.boxes.indices.first else { return }
                let box = row.boxes[boxIndex]
                let clampedX = min(max(current.x, box.left + 1), box.right - 1)
                manualDragState = .addAutoCut(rowID: rowID, boxIndex: boxIndex, currentX: clampedX)
            case .resizeAuto(let rowID, let corner, let startRect, _):
                manualDragState = .resizeAuto(rowID: rowID, corner: corner, startRect: startRect, currentPoint: location)
            }
            redrawOverlays()

        case .ended, .cancelled, .failed:
            defer {
                setManualInteractionLocked(false)
                manualDragState = nil
                redrawOverlays()
            }
            guard let drag = manualDragState else { return }
            switch drag {
            case .addRow(let page, let start, let current):
                guard let manualEditor = currentManualEditor,
                      manualEditor.tool == .addRow,
                      let row = manualRowRectFromPoints(page: page, start: start, end: current, staffKind: manualEditor.defaultStaffKind) else { return }
                appendManualRow(row)
                onManualSelectionChange?(ManualSelectionState(rowID: row.manualRowId, cutIndex: nil))
            case .addCut(let rowID, let currentX):
                guard var row = manualRow(for: rowID) else { return }
                let alreadyClose = row.cutXs.contains(where: { abs($0 - currentX) < 4 })
                guard !alreadyClose else { return }
                row.cutXs.append(currentX)
                row.cutXs = row.cutXs.sorted()
                replaceManualRow(row)
                if let cutIndex = row.cutXs.firstIndex(where: { abs($0 - currentX) < 0.001 }) {
                    onManualSelectionChange?(ManualSelectionState(rowID: row.manualRowId, cutIndex: cutIndex))
                }
            case .resize(let rowID, _, let startRect, let currentPoint):
                guard let row = manualRow(for: rowID),
                      let page = document.page(at: max(0, row.page - 1)),
                      let updated = previewRowRect(for: rowID, startRect: startRect, currentPoint: currentPoint, page: page, documentView: documentView) else { return }
                replaceManualRow(updated)
                onManualSelectionChange?(ManualSelectionState(rowID: rowID, cutIndex: nil))
            case .addAutoCut(let rowID, let boxIndex, let currentX):
                guard var row = autoRow(for: rowID),
                      boxIndex >= 0,
                      boxIndex < row.boxes.count else { return }
                let box = row.boxes[boxIndex]
                if abs(box.left - currentX) < 4 || abs(box.right - currentX) < 4 {
                    return
                }
                let newMeasureID = "auto_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(10))"
                let leftBox = AutoBoxState(
                    measureID: box.measureID,
                    left: box.left,
                    right: currentX,
                    excludedFromCounting: box.excludedFromCounting
                )
                let rightBox = AutoBoxState(
                    measureID: String(newMeasureID),
                    left: currentX,
                    right: box.right,
                    excludedFromCounting: box.excludedFromCounting
                )
                row.boxes.remove(at: boxIndex)
                row.boxes.insert(contentsOf: [leftBox, rightBox], at: boxIndex)
                replaceAutoRow(row)
                onAutoSelectionChange?(AutoSelectionState(rowID: row.systemID, splitIndex: nil, measureID: nil))
            case .resizeAuto(let rowID, _, let startRect, let currentPoint):
                guard let row = autoRow(for: rowID),
                      let page = document.page(at: max(0, row.page - 1)),
                      let updated = previewAutoRow(for: rowID, startRect: startRect, currentPoint: currentPoint, page: page, documentView: documentView) else { return }
                replaceAutoRow(updated)
                onAutoSelectionChange?(AutoSelectionState(rowID: row.systemID, splitIndex: nil, measureID: nil))
            }

        default:
            break
        }
    }

    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        if gestureRecognizer === panGesture || otherGestureRecognizer === panGesture {
            if manualDragState != nil {
                return false
            }
            if let pan = panGesture,
               let documentView = pdfView.documentView {
                let point = pan.location(in: documentView)
                if manualHandleHit(at: point, documentView: documentView) != nil
                    || autoHandleHit(at: point, documentView: documentView) != nil {
                    return false
                }
                switch currentManualEditor?.tool ?? currentAutoEditor?.tool {
                case .addRow:
                    return false
                case .addMeasures:
                    if let manualEditor = currentManualEditor,
                       let row = manualRowAtDocumentPoint(point, documentView: documentView),
                       row.page == manualEditor.activePage {
                        return false
                    }
                    if let autoEditor = currentAutoEditor,
                       let row = autoRowAtDocumentPoint(point, documentView: documentView),
                       row.page == autoEditor.activePage {
                        return false
                    }
                case .resizeRow, .delete, .exclude, .removeLabel, .none:
                    break
                }
            }
        }
        return true
    }

    override func gestureRecognizerShouldBegin(_ gestureRecognizer: UIGestureRecognizer) -> Bool {
        guard gestureRecognizer === panGesture,
              let pan = gestureRecognizer as? UIPanGestureRecognizer,
              let documentView = pdfView.documentView else {
            return true
        }
        let point = pan.location(in: documentView)
        if manualHandleHit(at: point, documentView: documentView) != nil
            || autoHandleHit(at: point, documentView: documentView) != nil {
            return true
        }
        let tool = currentManualEditor?.tool ?? currentAutoEditor?.tool
        guard let tool else {
            return true
        }
        switch tool {
        case .addRow:
            return true
        case .addMeasures:
            if let manualEditor = currentManualEditor,
               let row = manualRowAtDocumentPoint(point, documentView: documentView) {
                return row.page == manualEditor.activePage
            }
            if let autoEditor = currentAutoEditor,
               let row = autoRowAtDocumentPoint(point, documentView: documentView) {
                return row.page == autoEditor.activePage
            }
            return false
        case .resizeRow, .delete, .exclude, .removeLabel:
            return false
        }
    }
}

// MARK: - Settings Sheet

private struct SettingsSheet: View {
    @Binding var accentThemeRaw: String
    @Binding var forceDark: Bool
    @Environment(\.dismiss) private var dismiss

    private var accentTheme: AccentTheme { AccentTheme(rawValue: accentThemeRaw) ?? .orange }
    private let appVersion: String = {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "—"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "—"
        return "\(v) (\(b))"
    }()

    var body: some View {
        NavigationStack {
            List {
                // MARK: Account
                Section("Account") {
                    Label("Sign In with Apple", systemImage: "apple.logo")
                        .foregroundStyle(.primary)
                    Label("Restore Purchases", systemImage: "arrow.clockwise")
                        .foregroundStyle(.primary)
                }

                // MARK: Billing
                Section("Billing") {
                    Label("Subscription Status", systemImage: "creditcard")
                        .foregroundStyle(.primary)
                    Label("Upgrade Plan", systemImage: "star.fill")
                        .foregroundStyle(.primary)
                }

                // MARK: Promo Code
                Section("Promo Code") {
                    Label("Apply Code", systemImage: "tag.fill")
                        .foregroundStyle(.primary)
                }

                // MARK: Upload Usage
                Section("Upload Usage") {
                    Label("Remaining Uploads", systemImage: "icloud.and.arrow.up")
                        .foregroundStyle(.primary)
                }

                // MARK: Preferences
                Section("Preferences") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Color")
                            .font(.subheadline)
                        HStack(spacing: 12) {
                            ForEach(AccentTheme.allCases, id: \.rawValue) { theme in
                                Button {
                                    accentThemeRaw = theme.rawValue
                                } label: {
                                    VStack(spacing: 4) {
                                        Circle()
                                            .fill(theme == .classic ? Color(.systemBackground) : theme.color)
                                            .frame(width: 32, height: 32)
                                            .overlay(Circle().strokeBorder(Color.secondary.opacity(0.3), lineWidth: 1))
                                            .overlay(
                                                Circle()
                                                    .strokeBorder(accentTheme == theme ? Color.primary : Color.clear, lineWidth: 2.5)
                                                    .padding(2)
                                            )
                                        Text(theme.label)
                                            .font(.caption2)
                                            .foregroundStyle(accentTheme == theme ? .primary : .secondary)
                                    }
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    Toggle(isOn: $forceDark) {
                        Label("Dark Mode", systemImage: "moon.fill")
                    }
                }

                // MARK: Support
                Section("Support") {
                    Button {
                        let subject = "Sheet Music Labeler Feedback".addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
                        if let url = URL(string: "mailto:suggestions.pineapple@gmail.com?subject=\(subject)") {
                            UIApplication.shared.open(url)
                        }
                    } label: {
                        Label("Help & Feedback", systemImage: "envelope")
                    }
                    .foregroundStyle(.primary)

                    HStack {
                        Label("App Version", systemImage: "info.circle")
                        Spacer()
                        Text(appVersion)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .tint(accentTheme.color)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .preferredColorScheme(forceDark ? .dark : .light)
        }
    }
}

// MARK: - Document Picker

private final class DocumentPickerDelegate: NSObject, UIDocumentPickerDelegate {
    static let shared = DocumentPickerDelegate()
    var onPick: ((URL) -> Void)?

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        guard let url = urls.first else { return }
        onPick?(url)
    }
}

private struct LocalError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

private extension String {
    var nonEmpty: String? {
        let trimmed = trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

#Preview {
    ContentView()
}

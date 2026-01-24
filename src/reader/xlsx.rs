use std::io::Cursor;

use calamine::{DataType, Reader as CalamineReader, Xlsx};

use crate::{
    DocumentFormat, DocumentReader, PassthroughReader, ReaderDiagnostics, ReaderHint, ReaderOutput,
    Result,
};

pub struct XlsxReader;

impl XlsxReader {
    fn extract_text(bytes: &[u8]) -> Result<String> {
        let cursor = Cursor::new(bytes);
        let mut workbook =
            Xlsx::new(cursor).map_err(|err| crate::MemvidError::ExtractionFailed {
                reason: format!("failed to read xlsx workbook: {err}").into(),
            })?;

        let mut out = String::new();
        for sheet_name in workbook.sheet_names().clone() {
            if let Some(Ok(range)) = workbook.worksheet_range(&sheet_name) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!("Sheet: {sheet_name}\n"));
                for row in range.rows() {
                    let mut first_cell = true;
                    for cell in row {
                        if !first_cell {
                            out.push('\t');
                        }
                        first_cell = false;
                        match cell {
                            DataType::String(s) => out.push_str(s.trim()),
                            DataType::Float(v) => out.push_str(&format!("{v}")),
                            DataType::Int(v) => out.push_str(&format!("{v}")),
                            DataType::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
                            DataType::Error(e) => out.push_str(&format!("#{e:?}")),
                            DataType::Empty => {}
                            DataType::DateTime(v) => out.push_str(&format!("{v}")),
                            DataType::DateTimeIso(s) => out.push_str(s),
                            DataType::Duration(v) => out.push_str(&format!("{v}")),
                            DataType::DurationIso(s) => out.push_str(s),
                        }
                    }
                    out.push('\n');
                }
            }
        }

        Ok(out.trim().to_string())
    }
}

impl DocumentReader for XlsxReader {
    fn name(&self) -> &'static str {
        "xlsx"
    }

    fn supports(&self, hint: &ReaderHint<'_>) -> bool {
        matches!(hint.format, Some(DocumentFormat::Xlsx))
            || hint.mime.is_some_and(|mime| {
                mime.eq_ignore_ascii_case(
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            })
    }

    fn extract(&self, bytes: &[u8], hint: &ReaderHint<'_>) -> Result<ReaderOutput> {
        match Self::extract_text(bytes) {
            Ok(text) => {
                if text.trim().is_empty() {
                    // Calamine returned empty - try extractous as fallback
                    let mut fallback = PassthroughReader.extract(bytes, hint)?;
                    fallback.reader_name = self.name().to_string();
                    fallback.diagnostics.mark_fallback();
                    fallback.diagnostics.record_warning(
                        "xlsx reader produced empty text; falling back to default extractor",
                    );
                    Ok(fallback)
                } else {
                    // Calamine succeeded - build output directly WITHOUT calling extractous
                    let mut document = crate::ExtractedDocument::empty();
                    document.text = Some(text);
                    document.mime_type = Some(
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            .to_string(),
                    );
                    Ok(ReaderOutput::new(document, self.name())
                        .with_diagnostics(ReaderDiagnostics::default()))
                }
            }
            Err(err) => {
                // Calamine failed - try extractous as fallback
                let mut fallback = PassthroughReader.extract(bytes, hint)?;
                fallback.reader_name = self.name().to_string();
                fallback.diagnostics.mark_fallback();
                fallback
                    .diagnostics
                    .record_warning(format!("xlsx reader error: {err}"));
                Ok(fallback)
            }
        }
    }
}

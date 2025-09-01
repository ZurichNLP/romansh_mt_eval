import os
import pandas as pd

class WorksheetWriter:
    def __init__(self, df, cols, dir_name):
        self.df = df
        self.input_cols = cols
        self.output_cols = ['German' if col == 'target' else 'English' if col == 'source' else col for col in cols]
        self.dir_name = dir_name


    def create_worksheet(self, variety):
        with pd.ExcelWriter(os.path.join(self.dir_name, variety)+".xlsx", engine="xlsxwriter") as writer:
            workbook = writer.book
            # Create the cover sheet first (it will be the first sheet)
            cover_ws = workbook.add_worksheet("cover")

            domain_completion_info = []
            for domain, group in self.df.groupby("domain"):
                sheet_name = str(domain)[:31]  # Excel sheet names must be <=31 characters
                final_with_gaps = self.prepare_df(group)
                final_with_gaps.to_excel(writer, sheet_name=sheet_name, index=False, freeze_panes=(1, 0))
                self.restrict_editing(writer, sheet_name)
                info = self.collect_cover_sheet_info(len(final_with_gaps), sheet_name, domain)
                domain_completion_info.append(info)

            self.populate_cover_sheet(writer, cover_ws, variety, domain_completion_info)


    def collect_cover_sheet_info(self, total_rows, sheet_name, domain):
        start_row = 2  # Data starts from row 3 (1-based indexing, Excel)
        trans_col_letter = chr(ord('A') + self.output_cols.index("translation"))  # Convert column index to Excel letter
        target_col_letter = chr(ord('A') + self.output_cols.index("German"))  # Convert column index to Excel letter
        res = {
            "Domain": domain, 
            "Completion %": f'=COUNTIFS(\'{sheet_name}\'!{trans_col_letter}{start_row}:{trans_col_letter}{total_rows + 2},"?*", {sheet_name}!{target_col_letter}{start_row}:{target_col_letter}{total_rows + 2}, "<>")/COUNTA(\'{sheet_name}\'!{target_col_letter}{start_row}:{target_col_letter}{total_rows + 2})'
        }
        return res


    def populate_cover_sheet(self, writer, cover_ws, variety, domain_completion_info):
        cover_df = pd.DataFrame(domain_completion_info)
        workbook = writer.book
        cover_df.to_excel(writer, sheet_name="cover", index=False, startrow=3)
        cover_ws.write("A1", "Target variety:")
        cover_ws.write("B1", variety)
        locked_format = workbook.add_format({'locked': True})
        cover_ws.set_column(0, len(cover_df.columns), 50, locked_format)

        percent_format = workbook.add_format({'num_format': '0.00%', 'locked': True})
        try:
            percent_col_idx = cover_df.columns.get_loc("Completion %")
            cover_ws.set_column(percent_col_idx, percent_col_idx, 15, percent_format)
        except KeyError:
             print("Warning: 'Completion %' column not found in cover sheet data.")

        cover_ws.protect(options={'insert_rows': False, 'delete_rows': False, 'sort': False})


    def prepare_df(self, group):
        final_df = group[["source", "document_id", "segment_id", "url", "target"]].sort_values(by=["document_id", "segment_id"])
        final_df["translation"] = ""
        final_df["comment"] = ""
        final_df["doc_change"] = final_df["document_id"].ne(final_df["document_id"].shift())
        rows_with_gaps = []
        gap_indices = []  # Track indices of gap rows
        row_index = 0
        for _, row in final_df.iterrows():
            if row["doc_change"]:
                rows_with_gaps.append(pd.Series([None]*len(self.input_cols), index=self.input_cols))  # empty row
                gap_indices.append(row_index)
                row_index += 1
            rows_with_gaps.append(row[self.input_cols])  # actual row
            row_index += 1
        final_with_gaps = pd.DataFrame(rows_with_gaps)
        final_with_gaps = final_with_gaps.rename(columns={"target": "German", "source": "English"})
        # Store gap indices as an attribute for use in restrict_editing
        self.gap_indices = gap_indices
        return final_with_gaps
    

    def restrict_editing(self, writer, sheet_name):
        workbook  = writer.book
        worksheet = writer.sheets[sheet_name]
        cell_format_locked = workbook.add_format({'locked': True, 'valign': 'top'})
        cell_format_locked_wrap_top = workbook.add_format({'locked': True, 'text_wrap': True, 'valign': 'top'})
        unlock_format_wrap_top = workbook.add_format({'locked': False, 'text_wrap': True, 'valign': 'top'})
        worksheet.set_column(0, len(self.output_cols)-1, 40, cell_format_locked)

        trans_idx = self.output_cols.index("translation")
        comment_idx = self.output_cols.index("comment")
        german_idx = self.output_cols.index("German")
        english_idx = self.output_cols.index("English")
        document_id_idx = self.output_cols.index("document_id")
        segment_id_idx = self.output_cols.index("segment_id")
        url_idx = self.output_cols.index("url")

        worksheet.set_column(german_idx, german_idx, 60, cell_format_locked_wrap_top)
        worksheet.set_column(english_idx, english_idx, 60, cell_format_locked_wrap_top)
        worksheet.set_column(trans_idx, trans_idx, 60, unlock_format_wrap_top)
        worksheet.set_column(comment_idx, comment_idx, 20, unlock_format_wrap_top)
        worksheet.set_column(document_id_idx, document_id_idx, 10, cell_format_locked)
        worksheet.set_column(segment_id_idx, segment_id_idx, 10, cell_format_locked)
        worksheet.set_column(url_idx, url_idx, 10, cell_format_locked)

        # Add background color to gap rows
        gap_format = workbook.add_format({'bg_color': '#E0E0E0'})
        for gap_idx in self.gap_indices:
            # Excel rows are 1-indexed and we need to account for the header row
            excel_row = gap_idx + 1
            worksheet.set_row(excel_row, None, gap_format)

        worksheet.protect(options={'insert_rows': False, 'delete_rows': False, 'sort': False})

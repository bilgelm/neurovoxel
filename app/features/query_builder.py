import streamlit as st

def query_builder(image_names: list[str]) -> dict[str, str]:
	"""
	Display a query builder form with variable buttons.
	Clicking a variable button appends it to the query field.
	"""
	st.subheader("Query Builder")

	# Use session state to persist query string
	if "query_string" not in st.session_state:
		st.session_state["query_string"] = ""

	# Show variable buttons
	st.write("Click a variable to add it to your query:")
	cols = st.columns(min(len(image_names), 5))
	for idx, name in enumerate(image_names):
		if cols[idx % 5].button(name, key=f"varbtn_{name}"):
			# Only add variable if not already present as a distinct token
			import re
			current_query = st.session_state["query_string"]
			# Tokenize by space, +, ~, =, etc.
			tokens = re.split(r'[\s\+~=]+', current_query)
			if name not in tokens:
				if current_query and not current_query.endswith((' ', '+', '~', '=')):
					st.session_state["query_string"] += f" + {name}"
				else:
					st.session_state["query_string"] += name

	# Query field and Run button side by side
	# Query field
	query_string = st.text_input(
		"Build your query by clicking variables and/or typing:",
		value=st.session_state["query_string"],
		key="query_string_input"
	)

	# Advanced settings dropdown
	with st.expander("Advanced"):
		smoothing = st.number_input(
			"Smoothing FWHM (mm) for isotropic Gaussian",
			min_value=0.0,
			value=5.0,
			step=0.5,
			format="%.2f",
			key="smoothing_fwhm"
		)
		voxel_size = st.number_input(
			"Voxel size of statistical analysis space (mm, isotropic)",
			min_value=0.0,
			value=6.0,
			step=0.5,
			format="%.2f",
			key="voxel_size"
		)
		permutations = st.number_input(
			"Number of permutations",
			min_value=1,
			value=100,
			step=1,
			key="num_permutations"
		)
		run_tfce = st.checkbox("Run TFCE", key="run_tfce")

	# Run analysis button after advanced settings
	run_clicked = st.button("Run analysis", key="run_analysis_btn")

	# Sync session state with manual edits
	st.session_state["query_string"] = query_string

	if run_clicked:
		st.success(f"Analysis submitted: {query_string}")

	st.caption("Click variable buttons to add them to your query, or type manually. Then click 'Run analysis'.")
	return {
		"query": query_string,
		"run_clicked": run_clicked,
		"smoothing_fwhm": smoothing,
		"voxel_size": voxel_size,
		"num_permutations": permutations,
		"run_tfce": run_tfce
	}

import React from "react";
import { core } from "@apache-superset/core";

export const activate = (context: core.ExtensionContext) => {
  context.disposables.push(
    core.registerViewProvider("pandas_semantic_layer.example", () => <p>Pandas Semantic Layer</p>)
  );
  console.log("Pandas Semantic Layer extension activated");
};

export const deactivate = () => {
  console.log("Pandas Semantic Layer extension deactivated");
};
